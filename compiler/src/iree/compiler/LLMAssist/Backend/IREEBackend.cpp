// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Backend/IREEBackend.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"

#if defined(IREE_HAVE_HAL_HIP_DRIVER)
#include "iree/hal/drivers/hip/registration/driver_module.h"
#endif
#include "iree/io/file_handle.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

namespace {

/// Helper to format an IREE status to a string.
std::string formatStatus(iree_status_t status) {
  char buffer[256];
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(buffer), buffer, &length);
  return std::string(buffer, std::min(length, sizeof(buffer) - 1));
}

} // namespace

//===----------------------------------------------------------------------===//
// LLMModelConfig
//===----------------------------------------------------------------------===//

llvm::Expected<LLMModelConfig>
LLMModelConfig::loadFromFile(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    return llvm::createStringError(bufferOrErr.getError(),
                                   "Failed to open config file: %s",
                                   path.str().c_str());
  }

  auto json = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!json) {
    return json.takeError();
  }

  LLMModelConfig config;
  if (auto *obj = json->getAsObject()) {
    // Try to load from llm_assist nested section first (new format)
    if (auto *llmAssist = obj->getObject("llm_assist")) {
      if (auto v = llmAssist->getInteger("num_layers"))
        config.numLayers = *v;
      if (auto v = llmAssist->getInteger("num_heads"))
        config.numHeads = *v;
      if (auto v = llmAssist->getInteger("num_kv_heads"))
        config.numKVHeads = *v;
      if (auto v = llmAssist->getInteger("head_dim"))
        config.headDim = *v;
      if (auto v = llmAssist->getInteger("vocab_size"))
        config.vocabSize = *v;
      if (auto v = llmAssist->getInteger("block_seq_stride"))
        config.blockSeqStride = *v;
      if (auto v = llmAssist->getInteger("device_block_count"))
        config.deviceBlockCount = *v;
      if (auto v = llmAssist->getInteger("page_size"))
        config.pageSize = *v;
      if (auto v = llmAssist->getInteger("context_length"))
        config.contextLength = *v;
      if (auto s = llmAssist->getString("model_type"))
        config.modelType = s->str();
    }

    // Load from paged_kv_cache section (alternative source for some values)
    if (auto *kvCache = obj->getObject("paged_kv_cache")) {
      if (auto v = kvCache->getInteger("attention_head_count_kv"))
        config.numKVHeads = *v;
      if (auto v = kvCache->getInteger("block_seq_stride"))
        config.blockSeqStride = *v;
    }

    // Fallback to top-level keys (old format or missing llm_assist)
    if (auto v = obj->getInteger("transformer_block_count"))
      config.numLayers = *v;
    if (auto v = obj->getInteger("attn_head_dim"))
      config.headDim = *v;
    if (auto v = obj->getInteger("num_layers"))
      config.numLayers = *v;
    if (auto v = obj->getInteger("num_heads"))
      config.numHeads = *v;
    if (auto v = obj->getInteger("num_kv_heads"))
      config.numKVHeads = *v;
    if (auto v = obj->getInteger("head_dim"))
      config.headDim = *v;
    if (auto v = obj->getInteger("vocab_size"))
      config.vocabSize = *v;
    if (auto v = obj->getInteger("block_seq_stride"))
      config.blockSeqStride = *v;
    if (auto s = obj->getString("model_type"))
      config.modelType = s->str();
  }

  return config;
}

//===----------------------------------------------------------------------===//
// IREEBackend
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<IREEBackend>>
IREEBackend::create(const IREEBackendConfig &config) {
  auto backend = std::unique_ptr<IREEBackend>(new IREEBackend(config));

  if (auto err = backend->initialize()) {
    return std::move(err);
  }

  return backend;
}

IREEBackend::IREEBackend(const IREEBackendConfig &config)
    : config_(config), hostAllocator_(iree_allocator_system()) {}

IREEBackend::~IREEBackend() { cleanup(); }

void IREEBackend::cleanup() {
  // Release decode input views first, then buffers
  if (decodeTokenView_) {
    iree_hal_buffer_view_release(decodeTokenView_);
    decodeTokenView_ = nullptr;
  }
  if (decodeSeqLenView_) {
    iree_hal_buffer_view_release(decodeSeqLenView_);
    decodeSeqLenView_ = nullptr;
  }
  if (decodeStartPosView_) {
    iree_hal_buffer_view_release(decodeStartPosView_);
    decodeStartPosView_ = nullptr;
  }
  if (decodeTokenBuffer_) {
    iree_hal_buffer_release(decodeTokenBuffer_);
    decodeTokenBuffer_ = nullptr;
  }
  if (decodeSeqLenBuffer_) {
    iree_hal_buffer_release(decodeSeqLenBuffer_);
    decodeSeqLenBuffer_ = nullptr;
  }
  if (decodeStartPosBuffer_) {
    iree_hal_buffer_release(decodeStartPosBuffer_);
    decodeStartPosBuffer_ = nullptr;
  }
  if (decodePageTableBuffer_) {
    iree_hal_buffer_release(decodePageTableBuffer_);
    decodePageTableBuffer_ = nullptr;
  }
  
  // Release cache buffer view and buffer
  if (cacheBufferView_) {
    iree_hal_buffer_view_release(cacheBufferView_);
    cacheBufferView_ = nullptr;
  }
  if (cacheBuffer_) {
    iree_hal_buffer_release(cacheBuffer_);
    cacheBuffer_ = nullptr;
  }

  if (context_) {
    iree_vm_context_release(context_);
    context_ = nullptr;
  }
  if (llmModule_) {
    iree_vm_module_release(llmModule_);
    llmModule_ = nullptr;
  }
  if (paramsModule_) {
    iree_vm_module_release(paramsModule_);
    paramsModule_ = nullptr;
  }
  if (halModule_) {
    iree_vm_module_release(halModule_);
    halModule_ = nullptr;
  }
  if (device_) {
    iree_hal_device_release(device_);
    device_ = nullptr;
  }
  if (instance_) {
    iree_vm_instance_release(instance_);
    instance_ = nullptr;
  }

  vmfbBuffer_.reset();
  initialized_ = false;
}

llvm::Error IREEBackend::initialize() {
  llvm::errs() << "IREEBackend::initialize: Starting\n";
  cleanup();

  // Auto-detect config path from vmfb path if not specified
  std::string configPath = config_.configPath;
  if (configPath.empty() && !config_.vmfbPath.empty()) {
    // Look for config.json in same directory as VMFB
    llvm::SmallString<256> vmfbDir(config_.vmfbPath);
    llvm::sys::path::remove_filename(vmfbDir);
    llvm::sys::path::append(vmfbDir, "config.json");
    if (llvm::sys::fs::exists(vmfbDir)) {
      configPath = std::string(vmfbDir);
    }
  }

  // Load model config
  if (!configPath.empty()) {
    llvm::errs() << "IREEBackend::initialize: Loading model config from " 
                 << configPath << "\n";
    auto configOrErr = LLMModelConfig::loadFromFile(configPath);
    if (!configOrErr) {
      return configOrErr.takeError();
    }
    modelConfig_ = *configOrErr;
    llvm::errs() << "IREEBackend::initialize: Model config loaded - numLayers="
                 << modelConfig_.numLayers << ", numKVHeads="
                 << modelConfig_.numKVHeads << ", headDim="
                 << modelConfig_.headDim << ", blockSeqStride="
                 << modelConfig_.blockSeqStride << "\n";
  }

  // Initialize tokenizer (auto-detect type based on file extension)
  auto tokenizerOrErr = createTokenizer(config_.tokenizerPath);
  if (!tokenizerOrErr) {
    return tokenizerOrErr.takeError();
  }
  tokenizer_ = std::move(*tokenizerOrErr);
  llvm::errs() << "IREEBackend::initialize: Loaded " << tokenizer_->getName()
               << " tokenizer with vocab size " << tokenizer_->vocabSize()
               << "\n";

  // Create VM instance
  iree_status_t status =
      iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, hostAllocator_,
                              &instance_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create VM instance: %s",
                                   msg.c_str());
  }

  // Register HAL module types
  status = iree_hal_module_register_all_types(instance_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to register HAL types: %s",
                                   msg.c_str());
  }

  // Register drivers (ignore if already registered)
  status = iree_hal_local_task_driver_module_register(
      iree_hal_driver_registry_default());
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) != IREE_STATUS_ALREADY_EXISTS) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to register local-task driver: %s",
                                   msg.c_str());
  }
  iree_status_ignore(status);

#if defined(IREE_HAVE_HAL_HIP_DRIVER)
  llvm::errs() << "IREEBackend::initialize: Registering HIP driver\n";
  status =
      iree_hal_hip_driver_module_register(iree_hal_driver_registry_default());
  if (!iree_status_is_ok(status) &&
      iree_status_code(status) != IREE_STATUS_ALREADY_EXISTS) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to register HIP driver: %s",
                                   msg.c_str());
  }
  iree_status_ignore(status);
#endif

  // Create device
  llvm::errs() << "IREEBackend::initialize: Creating device '" << config_.deviceUri << "'\n";
  iree_string_view_t deviceUri =
      iree_make_cstring_view(config_.deviceUri.c_str());
  status = iree_hal_create_device(iree_hal_driver_registry_default(), deviceUri,
                                  hostAllocator_, &device_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create device '%s': %s",
                                   config_.deviceUri.c_str(), msg.c_str());
  }
  llvm::errs() << "IREEBackend::initialize: Device created successfully\n";

  // Create HAL module
  // API: iree_hal_module_create(instance, device_policy, device_count, devices,
  //                             flags, debug_sink, host_allocator, out_module)
  iree_hal_module_debug_sink_t debugSink = {};
  status = iree_hal_module_create(
      instance_, iree_hal_module_device_policy_default(),
      /*device_count=*/1, &device_, IREE_HAL_MODULE_FLAG_NONE, debugSink,
      hostAllocator_, &halModule_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create HAL module: %s",
                                   msg.c_str());
  }

  // Load parameters from IRPA file (if provided)
  if (!config_.irpaPath.empty()) {
    // Open the IRPA file
    iree_io_file_handle_t *fileHandle = nullptr;
    iree_string_view_t irpaPath = iree_make_string_view(
        config_.irpaPath.data(), config_.irpaPath.size());
    status = iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, irpaPath,
                                      hostAllocator_, &fileHandle);
    if (!iree_status_is_ok(status)) {
      auto msg = formatStatus(status);
      iree_status_free(status);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to open IRPA file '%s': %s",
                                     config_.irpaPath.c_str(), msg.c_str());
    }

    // Create a parameter index and parse the IRPA file into it
    iree_io_parameter_index_t *paramIndex = nullptr;
    status = iree_io_parameter_index_create(hostAllocator_, &paramIndex);
    if (!iree_status_is_ok(status)) {
      iree_io_file_handle_release(fileHandle);
      auto msg = formatStatus(status);
      iree_status_free(status);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to create parameter index: %s",
                                     msg.c_str());
    }

    status = iree_io_parse_file_index(irpaPath, fileHandle, paramIndex,
                                      hostAllocator_);
    iree_io_file_handle_release(fileHandle);
    if (!iree_status_is_ok(status)) {
      iree_io_parameter_index_release(paramIndex);
      auto msg = formatStatus(status);
      iree_status_free(status);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to parse IRPA file: %s",
                                     msg.c_str());
    }

    // Create a parameter provider from the index
    // Note: The model expects parameters with scope "model"
    iree_io_parameter_provider_t *paramProvider = nullptr;
    status = iree_io_parameter_index_provider_create(
        iree_make_cstring_view("model"), // scope must match model's parameter references
        paramIndex,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        hostAllocator_, &paramProvider);
    iree_io_parameter_index_release(paramIndex);
    if (!iree_status_is_ok(status)) {
      auto msg = formatStatus(status);
      iree_status_free(status);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to create parameter provider: %s",
                                     msg.c_str());
    }

    // Create the I/O parameters module
    status = iree_io_parameters_module_create(
        instance_, /*provider_count=*/1, &paramProvider, hostAllocator_,
        &paramsModule_);
    iree_io_parameter_provider_release(paramProvider);
    if (!iree_status_is_ok(status)) {
      auto msg = formatStatus(status);
      iree_status_free(status);
      return llvm::createStringError(std::errc::io_error,
                                     "Failed to create parameters module: %s",
                                     msg.c_str());
    }
  }

  // Load VMFB module
  auto vmfbBufferOrErr = llvm::MemoryBuffer::getFile(config_.vmfbPath);
  if (!vmfbBufferOrErr) {
    return llvm::createStringError(vmfbBufferOrErr.getError(),
                                   "Failed to open VMFB file: %s",
                                   config_.vmfbPath.c_str());
  }
  // Store the buffer so it stays alive for the module's lifetime
  vmfbBuffer_ = std::move(*vmfbBufferOrErr);

  iree_const_byte_span_t vmfbData = {
      reinterpret_cast<const uint8_t *>(vmfbBuffer_->getBufferStart()),
      vmfbBuffer_->getBufferSize()};

  status = iree_vm_bytecode_module_create(
      instance_, vmfbData, iree_allocator_null(), hostAllocator_, &llmModule_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to load VMFB module: %s",
                                   msg.c_str());
  }

  // Create context with all modules
  // Order: io_parameters first, then HAL, then the LLM module (matches Python)
  std::vector<iree_vm_module_t *> modules;
  if (paramsModule_) {
    modules.push_back(paramsModule_);
  }
  modules.push_back(halModule_);
  modules.push_back(llmModule_);

  status = iree_vm_context_create_with_modules(
      instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
      hostAllocator_, &context_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create VM context: %s",
                                   msg.c_str());
  }

  // Look up prefill and decode functions
  status = iree_vm_context_resolve_function(
      context_, iree_make_cstring_view("module.prefill_bs1"), &prefillFn_);
  if (!iree_status_is_ok(status)) {
    // Try alternative names
    iree_status_free(status);
    status = iree_vm_context_resolve_function(
        context_, iree_make_cstring_view("module.prefill"), &prefillFn_);
  }
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to find prefill function: %s",
                                   msg.c_str());
  }

  status = iree_vm_context_resolve_function(
      context_, iree_make_cstring_view("module.decode_bs1"), &decodeFn_);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    status = iree_vm_context_resolve_function(
        context_, iree_make_cstring_view("module.decode"), &decodeFn_);
  }
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to find decode function: %s",
                                   msg.c_str());
  }

  // Initialize flat KV-cache buffer (matches Python test approach)
  blockSeqStride_ = modelConfig_.blockSeqStride;
  deviceBlockCount_ = modelConfig_.deviceBlockCount;
  
  // Use page_size from config if provided, otherwise calculate
  if (modelConfig_.pageSize > 0) {
    pageSize_ = modelConfig_.pageSize;
  } else {
    // Default calculation: numLayers * 2 * numKVHeads * blockSeqStride * headDim
    pageSize_ = static_cast<int64_t>(modelConfig_.numLayers) * 2 *
                static_cast<int64_t>(modelConfig_.numKVHeads) *
                static_cast<int64_t>(blockSeqStride_) *
                static_cast<int64_t>(modelConfig_.headDim);
  }

  llvm::errs() << "IREEBackend::initialize: Initializing cache...\n";
  if (auto err = initializeCache()) {
    return err;
  }
  llvm::errs() << "IREEBackend::initialize: Cache initialized successfully\n";

  initialized_ = true;
  llvm::errs() << "IREEBackend::initialize: Complete!\n";
  return llvm::Error::success();
}

llvm::Error IREEBackend::initializeCache() {
  llvm::errs() << "IREEBackend::initializeCache: deviceBlockCount_=" << deviceBlockCount_
               << ", pageSize_=" << pageSize_ << "\n";
  llvm::errs().flush();
               
  // Allocate flat cache buffer: [deviceBlockCount_, pageSize_] of f16
  size_t cacheBytes = static_cast<size_t>(deviceBlockCount_) *
                      static_cast<size_t>(pageSize_) * sizeof(uint16_t);
                      
  llvm::errs() << "IREEBackend::initializeCache: cacheBytes=" << cacheBytes << "\n";
  llvm::errs().flush();

  llvm::errs() << "IREEBackend::initializeCache: Creating buffer params\n";
  llvm::errs().flush();
  
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  llvm::errs() << "IREEBackend::initializeCache: Allocating buffer\n";
  llvm::errs().flush();
  
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, cacheBytes, &cacheBuffer_);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate KV-cache buffer: %s",
                                   msg.c_str());
  }

  llvm::errs() << "IREEBackend::initializeCache: Buffer allocated\n";
  llvm::errs().flush();
  
  // Note: Skip zeroing - the cache will be initialized by the model
  // iree_hal_buffer_map_zero may not work with device-local GPU buffers

  llvm::errs() << "IREEBackend::initializeCache: Creating buffer view\n";
  llvm::errs().flush();
  
  // Create buffer view: [deviceBlockCount_, pageSize_]
  iree_hal_dim_t cacheDims[] = {static_cast<iree_hal_dim_t>(deviceBlockCount_),
                                static_cast<iree_hal_dim_t>(pageSize_)};
  status = iree_hal_buffer_view_create(
      cacheBuffer_, 2, cacheDims, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, hostAllocator_, &cacheBufferView_);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create cache buffer view: %s",
                                   msg.c_str());
  }

  // Pre-allocate decode input buffers for reuse
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device_);
  iree_hal_buffer_params_t inputParams = {};
  inputParams.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  inputParams.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  inputParams.access = IREE_HAL_MEMORY_ACCESS_ALL;
  
  // Token buffer: single i64
  status = iree_hal_allocator_allocate_buffer(allocator, inputParams, sizeof(int64_t), &decodeTokenBuffer_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate decode token buffer: %s", msg.c_str());
  }
  
  // SeqLen buffer: single i64
  status = iree_hal_allocator_allocate_buffer(allocator, inputParams, sizeof(int64_t), &decodeSeqLenBuffer_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate decode seq_len buffer: %s", msg.c_str());
  }
  
  // StartPos buffer: single i64
  status = iree_hal_allocator_allocate_buffer(allocator, inputParams, sizeof(int64_t), &decodeStartPosBuffer_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate decode start_pos buffer: %s", msg.c_str());
  }
  
  // Page table buffer: allocate for max expected pages (enough for context length)
  int maxPages = (2048 + blockSeqStride_ - 1) / blockSeqStride_;
  decodePageTableCapacity_ = maxPages;
  status = iree_hal_allocator_allocate_buffer(allocator, inputParams, maxPages * sizeof(int64_t), &decodePageTableBuffer_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::not_enough_memory,
                                   "Failed to allocate decode page_table buffer: %s", msg.c_str());
  }
  
  // Create buffer views for fixed-size buffers (token [1,1], seq_len [1], start_pos [1])
  const iree_hal_dim_t tokenDims[] = {1, 1};
  const iree_hal_dim_t scalarDims[] = {1};
  
  status = iree_hal_buffer_view_create(decodeTokenBuffer_, 2, tokenDims,
                                        IREE_HAL_ELEMENT_TYPE_INT_64,
                                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                        hostAllocator_, &decodeTokenView_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create token view: %s", msg.c_str());
  }
  
  status = iree_hal_buffer_view_create(decodeSeqLenBuffer_, 1, scalarDims,
                                        IREE_HAL_ELEMENT_TYPE_INT_64,
                                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                        hostAllocator_, &decodeSeqLenView_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create seq_len view: %s", msg.c_str());
  }
  
  status = iree_hal_buffer_view_create(decodeStartPosBuffer_, 1, scalarDims,
                                        IREE_HAL_ELEMENT_TYPE_INT_64,
                                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                        hostAllocator_, &decodeStartPosView_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create start_pos view: %s", msg.c_str());
  }

  llvm::errs() << "IREEBackend::initializeCache: Done (with decode buffers and views)\n";
  llvm::errs().flush();
  
  return llvm::Error::success();
}

bool IREEBackend::isAvailable() const { return initialized_; }

llvm::Expected<GenerationResult>
IREEBackend::generate(llvm::StringRef prompt, const GenerationConfig &config) {
  llvm::errs() << "IREEBackend::generate: Starting\n";
  
  if (!initialized_) {
    return llvm::createStringError(std::errc::operation_not_permitted,
                                   "IREE backend not initialized");
  }

  auto startTime = std::chrono::steady_clock::now();

  llvm::errs() << "IREEBackend::generate: Tokenizing prompt\n";
  
  // Tokenize input
  std::vector<int64_t> inputTokens = tokenizer_->encode(prompt);
  llvm::errs() << "IREEBackend::generate: Got " << inputTokens.size() << " tokens\n";

  // Add BOS token if not present
  if (inputTokens.empty() || inputTokens[0] != tokenizer_->bosId()) {
    inputTokens.insert(inputTokens.begin(), tokenizer_->bosId());
  }

  // Re-initialize cache for new generation (zero it out)
  if (cacheBuffer_) {
    size_t cacheBytes = static_cast<size_t>(deviceBlockCount_) *
                        static_cast<size_t>(pageSize_) * sizeof(uint16_t);
    iree_status_t status = iree_hal_buffer_map_zero(cacheBuffer_, 0, cacheBytes);
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      // Non-fatal, continue anyway
    }
  }

  // Generate tokens
  int maxNewTokens =
      config.maxTokens > 0 ? config.maxTokens : config_.maxNewTokens;
  auto generatedTokensOrErr = generateTokens(inputTokens, maxNewTokens);
  if (!generatedTokensOrErr) {
    return generatedTokensOrErr.takeError();
  }

  // Decode output
  std::string output = tokenizer_->decode(*generatedTokensOrErr);

  auto endTime = std::chrono::steady_clock::now();
  auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        endTime - startTime)
                        .count();

  GenerationResult result;
  result.content = std::move(output);
  result.promptTokens = static_cast<int>(inputTokens.size());
  result.completionTokens =
      static_cast<int>(generatedTokensOrErr->size()) - result.promptTokens;
  result.latencyMs = static_cast<float>(durationMs);

  return result;
}

llvm::Expected<std::vector<int64_t>>
IREEBackend::generateTokens(llvm::ArrayRef<int64_t> promptTokens,
                            int maxNewTokens) {
  std::vector<int64_t> allTokens(promptTokens.begin(), promptTokens.end());
  int seqLen = static_cast<int>(promptTokens.size());

  // Prefill phase
  auto nextTokenOrErr = runPrefill(promptTokens, seqLen);
  if (!nextTokenOrErr) {
    return nextTokenOrErr.takeError();
  }

  int64_t nextToken = *nextTokenOrErr;
  allTokens.push_back(nextToken);

  // Decode loop
  int currentSeqLen = seqLen + 1; // After prefill, we've added one token
  // Use model's context length from config (2048 for this model)
  int maxSeqLen = modelConfig_.contextLength;
  
  for (int i = 0; i < maxNewTokens - 1; ++i) {
    // Check for EOS
    if (nextToken == tokenizer_->eosId()) {
      llvm::errs() << "IREEBackend::generateTokens: Hit EOS token\n";
      break;
    }
    
    // Check for context length limit
    if (currentSeqLen >= maxSeqLen) {
      llvm::errs() << "IREEBackend::generateTokens: Hit max sequence length "
                   << maxSeqLen << "\n";
      break;
    }

    // Decode next token
    // startPos = currentSeqLen - 1 (position of the token we just added)
    nextTokenOrErr = runDecode(nextToken, currentSeqLen, currentSeqLen - 1);
    if (!nextTokenOrErr) {
      return nextTokenOrErr.takeError();
    }

    nextToken = *nextTokenOrErr;
    allTokens.push_back(nextToken);
    ++currentSeqLen;
  }

  llvm::errs() << "IREEBackend::generateTokens: Generated " 
               << allTokens.size() - seqLen << " tokens\n";
  return allTokens;
}

// Set to true to enable detailed prefill/decode debug output
static const bool kEnableDebugOutput = true;

llvm::Expected<int64_t>
IREEBackend::runPrefill(llvm::ArrayRef<int64_t> tokens, int seqLen) {
  auto prefillStart = std::chrono::high_resolution_clock::now();
  if (kEnableDebugOutput)
    llvm::errs() << "IREEBackend::runPrefill: Starting with seqLen=" << seqLen << "\n";
  
  iree_status_t status = iree_ok_status();
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device_);
               
  // Model signature: prefill_bs1(tokens, seq_lens, page_table, cache)
  // tokens: [1, seq_len] i64
  // seq_lens: [1] i64
  // page_table: [1, num_pages] i64
  // cache: [block_count, page_size] f16

  // 1. Create tokens buffer: [1, seq_len]
  std::vector<int64_t> tokensCopy(tokens.begin(), tokens.end());
  iree_hal_buffer_view_t *tokensView = nullptr;
  const iree_hal_dim_t tokensDims[] = {1, static_cast<iree_hal_dim_t>(seqLen)};
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device_, allocator, 2, tokensDims, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      },
      iree_make_const_byte_span(tokensCopy.data(), seqLen * sizeof(int64_t)),
      &tokensView);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create tokens buffer: %s",
                                   msg.c_str());
  }

  // 2. Create seq_lens buffer: [1]
  int64_t seqLenVal = static_cast<int64_t>(seqLen);
  iree_hal_buffer_view_t *seqLenView = nullptr;
  const iree_hal_dim_t seqLenDims[] = {1};
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device_, allocator, 1, seqLenDims, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      },
      iree_make_const_byte_span(&seqLenVal, sizeof(int64_t)), &seqLenView);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(tokensView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create seq_lens buffer: %s",
                                   msg.c_str());
  }

  // 3. Create page_table buffer: [1, num_pages]
  int numPages = (seqLen + blockSeqStride_ - 1) / blockSeqStride_;
  std::vector<int64_t> pageTable(numPages);
  for (int i = 0; i < numPages; ++i) {
    pageTable[i] = i;
  }
  iree_hal_buffer_view_t *pageTableView = nullptr;
  const iree_hal_dim_t pageTableDims[] = {1, static_cast<iree_hal_dim_t>(numPages)};
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device_, allocator, 2, pageTableDims, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      },
      iree_make_const_byte_span(pageTable.data(), numPages * sizeof(int64_t)),
      &pageTableView);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(tokensView);
    iree_hal_buffer_view_release(seqLenView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create page_table buffer: %s",
                                   msg.c_str());
  }

  // 4. Build input list: tokens, seq_lens, page_table, cache
  iree_vm_list_t *inputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 4,
                               hostAllocator_, &inputs);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(tokensView);
    iree_hal_buffer_view_release(seqLenView);
    iree_hal_buffer_view_release(pageTableView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create input list: %s",
                                   msg.c_str());
  }

  iree_vm_ref_t tokensRef = iree_hal_buffer_view_move_ref(tokensView);
  iree_vm_ref_t seqLenRef = iree_hal_buffer_view_move_ref(seqLenView);
  iree_vm_ref_t pageTableRef = iree_hal_buffer_view_move_ref(pageTableView);
  iree_vm_ref_t cacheRef = iree_hal_buffer_view_retain_ref(cacheBufferView_);

  iree_vm_list_push_ref_move(inputs, &tokensRef);
  iree_vm_list_push_ref_move(inputs, &seqLenRef);
  iree_vm_list_push_ref_move(inputs, &pageTableRef);
  iree_vm_list_push_ref_move(inputs, &cacheRef);

  // 5. Create output list
  iree_vm_list_t *outputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                               hostAllocator_, &outputs);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(inputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create output list: %s",
                                   msg.c_str());
  }

  // 6. Invoke prefill
  auto allocEnd = std::chrono::high_resolution_clock::now();
  auto invokeStart = std::chrono::high_resolution_clock::now();
  status = iree_vm_invoke(context_, prefillFn_, IREE_VM_INVOCATION_FLAG_NONE,
                          nullptr, inputs, outputs, hostAllocator_);
  auto invokeEnd = std::chrono::high_resolution_clock::now();
  iree_vm_list_release(inputs);

  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    llvm::errs() << "IREEBackend::runPrefill: Invoke failed: " << msg << "\n";
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to invoke prefill: %s", msg.c_str());
  }

  // 7. Extract output tokens from result[1] (not result[0]!)
  // Output is [1, block_stride, 1] i64, token at position (seqLen - 1)
  llvm::errs() << "IREEBackend::runPrefill: Getting output list size\n";
  llvm::errs().flush();
  
  iree_host_size_t outputSize = iree_vm_list_size(outputs);
  llvm::errs() << "IREEBackend::runPrefill: Output list has " << outputSize << " elements\n";
  llvm::errs().flush();
  
  if (outputSize < 2) {
    iree_vm_list_release(outputs);
    return llvm::createStringError(std::errc::io_error,
                                   "Prefill output has less than 2 elements");
  }
  
  iree_vm_ref_t tokensOutRef = iree_vm_ref_null();
  status = iree_vm_list_get_ref_assign(outputs, 1, &tokensOutRef);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to get tokens output: %s",
                                   msg.c_str());
  }

  llvm::errs() << "IREEBackend::runPrefill: Got output ref, dereferencing\n";
  llvm::errs().flush();
  
  iree_hal_buffer_view_t *tokensOutView =
      iree_hal_buffer_view_deref(tokensOutRef);
  if (!tokensOutView) {
    iree_vm_list_release(outputs);
    return llvm::createStringError(std::errc::io_error,
                                   "Tokens output is not a buffer view");
  }

  llvm::errs() << "IREEBackend::runPrefill: Getting output buffer info\n";
  llvm::errs().flush();
  
  // Print output buffer view shape
  iree_host_size_t rank = iree_hal_buffer_view_shape_rank(tokensOutView);
  llvm::errs() << "IREEBackend::runPrefill: Output rank = " << rank << ", shape = [";
  for (iree_host_size_t i = 0; i < rank; ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << iree_hal_buffer_view_shape_dim(tokensOutView, i);
  }
  llvm::errs() << "]\n";
  
  iree_hal_element_type_t elemType = iree_hal_buffer_view_element_type(tokensOutView);
  llvm::errs() << "IREEBackend::runPrefill: Element type = " << elemType << "\n";
  llvm::errs().flush();
  
  iree_hal_buffer_t *tokensOutBuffer =
      iree_hal_buffer_view_buffer(tokensOutView);
  iree_device_size_t tokensOutSize =
      iree_hal_buffer_view_byte_length(tokensOutView);
  
  llvm::errs() << "IREEBackend::runPrefill: Output buffer size = " << tokensOutSize << " bytes\n";
  llvm::errs().flush();

  // Read from device buffer - transfer directly to host memory
  size_t numElements = tokensOutSize / sizeof(int64_t);
  std::vector<int64_t> outputTokens(numElements);
  
  llvm::errs() << "IREEBackend::runPrefill: Copying " << numElements << " elements to host memory\n";
  llvm::errs().flush();
  
  // Copy from device to host memory
  status = iree_hal_device_transfer_d2h(
      device_, tokensOutBuffer, /*source_offset=*/0,
      /*target_buffer=*/outputTokens.data(),
      /*data_length=*/tokensOutSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to copy output to host: %s",
                                   msg.c_str());
  }

  // Print first few elements to debug
  llvm::errs() << "IREEBackend::runPrefill: First 10 output elements: [";
  for (size_t i = 0; i < std::min(numElements, size_t(10)); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << outputTokens[i];
  }
  llvm::errs() << "]\n";
  
  llvm::errs() << "IREEBackend::runPrefill: Elements around seqLen-1 (" << (seqLen - 1) << "): [";
  for (int i = std::max(0, seqLen - 5); i < std::min((int)numElements, seqLen + 5); ++i) {
    if (i > std::max(0, seqLen - 5)) llvm::errs() << ", ";
    llvm::errs() << outputTokens[i];
  }
  llvm::errs() << "]\n";
  llvm::errs().flush();

  // Output is [1, block_stride, 1], get token at position (seqLen - 1)
  int64_t nextToken = outputTokens[seqLen - 1];

  llvm::errs() << "IREEBackend::runPrefill: Got next token = " << nextToken << "\n";
  llvm::errs().flush();
  
  // Debug: Check cache at different offsets
  {
    uint16_t cacheCheck[8] = {0};
    // Check at offset 0
    iree_hal_device_transfer_d2h(
        device_, iree_hal_buffer_view_buffer(cacheBufferView_), 0,
        cacheCheck, 8 * sizeof(uint16_t),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    llvm::errs() << "IREEBackend::runPrefill: Cache[0:8]: ";
    for (int i = 0; i < 8; ++i) llvm::errs() << cacheCheck[i] << " ";
    llvm::errs() << "\n";
    
    // Check at offset = pageSize_ / 2 (middle of first page)
    size_t midOffset = (pageSize_ / 4) * sizeof(uint16_t);
    iree_hal_device_transfer_d2h(
        device_, iree_hal_buffer_view_buffer(cacheBufferView_), midOffset,
        cacheCheck, 8 * sizeof(uint16_t),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    llvm::errs() << "IREEBackend::runPrefill: Cache[mid]: ";
    for (int i = 0; i < 8; ++i) llvm::errs() << cacheCheck[i] << " ";
    llvm::errs() << "\n";
  }
  
  // Debug: Check result[0] from prefill
  {
    iree_vm_ref_t result0Ref = iree_vm_ref_null();
    if (iree_status_is_ok(iree_vm_list_get_ref_assign(outputs, 0, &result0Ref))) {
      iree_hal_buffer_view_t *result0View = iree_hal_buffer_view_deref(result0Ref);
      if (result0View) {
        iree_device_size_t byteLen = iree_hal_buffer_view_byte_length(result0View);
        llvm::errs() << "IREEBackend::runPrefill: result[0] bytes=" << byteLen;
        
        // Read a sample of result[0] to check for NaN
        if (byteLen >= 2) {
          uint16_t sample[4] = {0};
          iree_hal_device_transfer_d2h(device_, iree_hal_buffer_view_buffer(result0View),
                                        0, sample, std::min(byteLen, iree_device_size_t(8)),
                                        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                        iree_infinite_timeout());
          llvm::errs() << ", first f16 values: ";
          for (int i = 0; i < std::min(4, (int)(byteLen/2)); ++i) {
            llvm::errs() << llvm::format("0x%04x ", sample[i]);
          }
        }
        llvm::errs() << "\n";
      }
    }
  }
  
  iree_vm_list_release(outputs);

  auto prefillEnd = std::chrono::high_resolution_clock::now();
  auto allocMs = std::chrono::duration<double, std::milli>(allocEnd - prefillStart).count();
  auto invokeMs = std::chrono::duration<double, std::milli>(invokeEnd - invokeStart).count();
  auto totalMs = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();
  llvm::errs() << "IREEBackend::runPrefill timing: alloc=" << llvm::format("%.1f", allocMs)
               << "ms, invoke=" << llvm::format("%.1f", invokeMs)
               << "ms, total=" << llvm::format("%.1f", totalMs) << "ms\n";

  return nextToken;
}

llvm::Expected<int64_t> IREEBackend::runDecode(int64_t token, int seqLen,
                                               int startPos) {
  static int decodeCallCount = 0;
  auto decodeStart = std::chrono::high_resolution_clock::now();
  
  if (decodeCallCount < 2) {
    llvm::errs() << "IREEBackend::runDecode[" << decodeCallCount << "]: "
                 << "token=" << token << ", seqLen=" << seqLen << ", startPos=" << startPos << "\n";
  }
  decodeCallCount++;
  
  iree_status_t status = iree_ok_status();

  // Model signature: decode_bs1(tokens, seq_lens, start_positions, page_table, cache)
  // Use pre-allocated buffers and just update their contents via h2d transfer
  
  // 1. Update token buffer (reusing pre-allocated buffer)
  int64_t tokenVal = token;
  status = iree_hal_device_transfer_h2d(device_, &tokenVal, decodeTokenBuffer_,
                                         0, sizeof(int64_t),
                                         IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                         iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to update token buffer: %s", msg.c_str());
  }

  // 2. Update seq_len buffer  
  int64_t seqLenVal = static_cast<int64_t>(seqLen);
  status = iree_hal_device_transfer_h2d(device_, &seqLenVal, decodeSeqLenBuffer_,
                                         0, sizeof(int64_t),
                                         IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                         iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to update seq_len buffer: %s", msg.c_str());
  }

  // 3. Update start_pos buffer
  int64_t startPosVal = static_cast<int64_t>(startPos);
  status = iree_hal_device_transfer_h2d(device_, &startPosVal, decodeStartPosBuffer_,
                                         0, sizeof(int64_t),
                                         IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                         iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to update start_pos buffer: %s", msg.c_str());
  }

  // 4. Update page_table buffer
  int numPages = (seqLen + blockSeqStride_ - 1) / blockSeqStride_;
  std::vector<int64_t> pageTable(numPages);
  for (int i = 0; i < numPages; ++i) {
    pageTable[i] = i;
  }
  
  // Debug: Print decode inputs for first 2 calls
  if (decodeCallCount <= 2) {
    llvm::errs() << "IREEBackend::runDecode[" << (decodeCallCount-1) << "] INPUTS:\n";
    llvm::errs() << "  token=" << tokenVal << " (shape [1,1])\n";
    llvm::errs() << "  seq_len=" << seqLenVal << " (shape [1])\n";
    llvm::errs() << "  start_pos=" << startPosVal << " (shape [1])\n";
    llvm::errs() << "  page_table=[";
    for (int i = 0; i < std::min(numPages, 5); ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << pageTable[i];
    }
    if (numPages > 5) llvm::errs() << ", ...";
    llvm::errs() << "] (shape [1," << numPages << "])\n";
    llvm::errs() << "  blockSeqStride_=" << blockSeqStride_ << "\n";
  }
  
  status = iree_hal_device_transfer_h2d(device_, pageTable.data(), decodePageTableBuffer_,
                                         0, numPages * sizeof(int64_t),
                                         IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                         iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to update page_table buffer: %s", msg.c_str());
  }

  auto allocEnd = std::chrono::high_resolution_clock::now();
  
  // Create page_table buffer view with correct size (only this changes per call)
  // Token, seq_len, and start_pos views are reused from initialization
  iree_hal_buffer_view_t *pageTableView = nullptr;
  const iree_hal_dim_t pageTableDims[] = {1, static_cast<iree_hal_dim_t>(numPages)};
  
  status = iree_hal_buffer_view_create(decodePageTableBuffer_, 2, pageTableDims,
                                        IREE_HAL_ELEMENT_TYPE_INT_64,
                                        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                        hostAllocator_, &pageTableView);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create page_table view: %s", msg.c_str());
  }
  
  // 5. Build input list: tokens, seq_lens, start_positions, page_table, cache
  iree_vm_list_t *inputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 5,
                               hostAllocator_, &inputs);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(pageTableView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create input list: %s",
                                   msg.c_str());
  }

  // Use retain_ref for cached views, move_ref for per-call pageTable
  iree_vm_ref_t tokenRef = iree_hal_buffer_view_retain_ref(decodeTokenView_);
  iree_vm_ref_t seqLenRef = iree_hal_buffer_view_retain_ref(decodeSeqLenView_);
  iree_vm_ref_t startPosRef = iree_hal_buffer_view_retain_ref(decodeStartPosView_);
  iree_vm_ref_t pageTableRef = iree_hal_buffer_view_move_ref(pageTableView);
  iree_vm_ref_t cacheRef = iree_hal_buffer_view_retain_ref(cacheBufferView_);
  
  // Debug: print cache buffer view info  
  static bool printedCacheInfo = false;
  if (!printedCacheInfo) {
    iree_host_size_t cacheRank = iree_hal_buffer_view_shape_rank(cacheBufferView_);
    llvm::errs() << "IREEBackend::runDecode: Cache shape=[";
    for (iree_host_size_t i = 0; i < cacheRank; ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << iree_hal_buffer_view_shape_dim(cacheBufferView_, i);
    }
    llvm::errs() << "], elem_type=" << iree_hal_buffer_view_element_type(cacheBufferView_)
                 << ", bytes=" << iree_hal_buffer_view_byte_length(cacheBufferView_) << "\n";
    printedCacheInfo = true;
  }

  iree_vm_list_push_ref_move(inputs, &tokenRef);
  iree_vm_list_push_ref_move(inputs, &seqLenRef);
  iree_vm_list_push_ref_move(inputs, &startPosRef);
  iree_vm_list_push_ref_move(inputs, &pageTableRef);
  iree_vm_list_push_ref_move(inputs, &cacheRef);

  // 6. Create output list
  iree_vm_list_t *outputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                               hostAllocator_, &outputs);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(inputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create output list: %s",
                                   msg.c_str());
  }

  // 7. Invoke decode
  auto invokeStart = std::chrono::high_resolution_clock::now();
  status = iree_vm_invoke(context_, decodeFn_, IREE_VM_INVOCATION_FLAG_NONE,
                          nullptr, inputs, outputs, hostAllocator_);
  auto invokeEnd = std::chrono::high_resolution_clock::now();
  iree_vm_list_release(inputs);

  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to invoke decode: %s", msg.c_str());
  }

  // Debug: Check both result[0] and result[1] contents
  static bool printedResultInfo = false;
  if (!printedResultInfo) {
    iree_host_size_t outSize = iree_vm_list_size(outputs);
    llvm::errs() << "IREEBackend::runDecode: Output list has " << outSize << " elements\n";
    
    for (iree_host_size_t idx = 0; idx < outSize && idx < 2; ++idx) {
      iree_vm_ref_t resultRef = iree_vm_ref_null();
      if (iree_status_is_ok(iree_vm_list_get_ref_assign(outputs, idx, &resultRef))) {
        iree_hal_buffer_view_t *resultView = iree_hal_buffer_view_deref(resultRef);
        if (resultView) {
          iree_host_size_t rank = iree_hal_buffer_view_shape_rank(resultView);
          iree_hal_element_type_t elemType = iree_hal_buffer_view_element_type(resultView);
          iree_device_size_t byteLen = iree_hal_buffer_view_byte_length(resultView);
          llvm::errs() << "IREEBackend::runDecode: result[" << idx << "] rank=" << rank 
                       << ", shape=[";
          for (iree_host_size_t i = 0; i < rank; ++i) {
            if (i > 0) llvm::errs() << ", ";
            llvm::errs() << iree_hal_buffer_view_shape_dim(resultView, i);
          }
          llvm::errs() << "], elem_type=" << elemType << ", bytes=" << byteLen << "\n";
          
          // Try to read value from result[0] as well
          if (idx == 0 && byteLen <= 16) {
            uint8_t data[16] = {0};
            iree_hal_device_transfer_d2h(device_, iree_hal_buffer_view_buffer(resultView),
                                          0, data, byteLen,
                                          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                          iree_infinite_timeout());
            llvm::errs() << "IREEBackend::runDecode: result[0] raw bytes: ";
            for (size_t i = 0; i < byteLen; ++i) {
              llvm::errs() << llvm::format("%02x ", data[i]);
            }
            llvm::errs() << "\n";
          }
        }
      }
    }
    printedResultInfo = true;
  }

  // 8. Extract output token from result[1]
  // Output is [1, 1, 1] i64
  iree_vm_ref_t tokensOutRef = iree_vm_ref_null();
  status = iree_vm_list_get_ref_assign(outputs, 1, &tokensOutRef);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to get tokens output: %s",
                                   msg.c_str());
  }

  iree_hal_buffer_view_t *tokensOutView =
      iree_hal_buffer_view_deref(tokensOutRef);
  if (!tokensOutView) {
    iree_vm_list_release(outputs);
    return llvm::createStringError(std::errc::io_error,
                                   "Tokens output is not a buffer view");
  }

  // Print output buffer view shape (only once to reduce noise)
  static bool printedDecodeShape = false;
  if (!printedDecodeShape) {
    iree_host_size_t rank = iree_hal_buffer_view_shape_rank(tokensOutView);
    llvm::errs() << "IREEBackend::runDecode: Output rank = " << rank << ", shape = [";
    for (iree_host_size_t i = 0; i < rank; ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << iree_hal_buffer_view_shape_dim(tokensOutView, i);
    }
    iree_hal_element_type_t elemType = iree_hal_buffer_view_element_type(tokensOutView);
    iree_device_size_t byteLen = iree_hal_buffer_view_byte_length(tokensOutView);
    llvm::errs() << "], element_type=" << elemType << ", byte_length=" << byteLen << "\n";
    
    // Also check element type of expected i64 (0x10000040 = 268435520)
    llvm::errs() << "IREEBackend::runDecode: Expected i64 element type = " << IREE_HAL_ELEMENT_TYPE_INT_64 << "\n";
    printedDecodeShape = true;
  }

  // Read from device buffer - transfer directly to host memory
  iree_hal_buffer_t *tokensOutBuffer =
      iree_hal_buffer_view_buffer(tokensOutView);
  iree_device_size_t tokensOutSize =
      iree_hal_buffer_view_byte_length(tokensOutView);

  size_t numElements = tokensOutSize / sizeof(int64_t);
  std::vector<int64_t> outputTokens(numElements);
  
  // Copy from device to host memory
  status = iree_hal_device_transfer_d2h(
      device_, tokensOutBuffer, /*source_offset=*/0,
      /*target_buffer=*/outputTokens.data(),
      /*data_length=*/tokensOutSize,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to copy decode output to host: %s",
                                   msg.c_str());
  }

  // Print first few output elements (only first few decodes)
  static int decodeCount = 0;
  if (decodeCount < 3) {
    llvm::errs() << "IREEBackend::runDecode[" << decodeCount << "]: Output elements: [";
    for (size_t i = 0; i < std::min(numElements, size_t(5)); ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << outputTokens[i];
    }
    llvm::errs() << "]\n";
    
    // Print raw bytes
    llvm::errs() << "IREEBackend::runDecode[" << decodeCount << "]: Raw bytes: ";
    uint8_t *rawBytes = reinterpret_cast<uint8_t*>(outputTokens.data());
    for (size_t i = 0; i < std::min(tokensOutSize, iree_device_size_t(16)); ++i) {
      llvm::errs() << llvm::format("%02x ", rawBytes[i]);
    }
    llvm::errs() << "\n";
  }
  decodeCount++;

  // Output is [1, 1, 1], get the single token
  int64_t nextToken = outputTokens[0];
  
  auto decodeEnd = std::chrono::high_resolution_clock::now();
  
  // Print timing for first few decodes
  if (decodeCount <= 5) {
    auto allocMs = std::chrono::duration<double, std::milli>(allocEnd - decodeStart).count();
    auto invokeMs = std::chrono::duration<double, std::milli>(invokeEnd - invokeStart).count();
    auto totalMs = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
    llvm::errs() << "IREEBackend::runDecode timing: alloc=" << llvm::format("%.1f", allocMs)
                 << "ms, invoke=" << llvm::format("%.1f", invokeMs)
                 << "ms, total=" << llvm::format("%.1f", totalMs)
                 << "ms, token=" << nextToken << "\n";
  }

  iree_vm_list_release(outputs);

  return nextToken;
}

std::unique_ptr<LLMBackend>
createIREEBackend(const IREEBackendConfig &config) {
  auto backendOrErr = IREEBackend::create(config);
  if (!backendOrErr) {
    llvm::errs() << "Failed to create IREE backend: "
                 << llvm::toString(backendOrErr.takeError()) << "\n";
    return nullptr;
  }
  return std::move(*backendOrErr);
}

} // namespace mlir::iree_compiler::LLMAssist
