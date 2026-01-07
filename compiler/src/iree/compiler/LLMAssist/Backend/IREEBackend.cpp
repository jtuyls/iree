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

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <vector>

#define DEBUG_TYPE "iree-llm-backend"

namespace mlir::iree_compiler::LLMAssist {

namespace {

/// Convert IREE status to string for error messages.
std::string formatStatus(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char *message = nullptr;
  iree_host_size_t length = 0;
  if (iree_status_to_string(status, &allocator, &message, &length)) {
    std::string result(message, length);
    iree_allocator_free(allocator, message);
    return result;
  }
  return "Unknown error";
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
    if (auto v = obj->getInteger("num_layers"))
      config.numLayers = *v;
    if (auto v = obj->getInteger("num_kv_heads"))
      config.numKVHeads = *v;
    if (auto v = obj->getInteger("head_dim"))
      config.headDim = *v;
    if (auto v = obj->getInteger("vocab_size"))
      config.vocabSize = *v;
    if (auto v = obj->getInteger("prefill_len"))
      config.prefillLen = *v;
    if (auto v = obj->getInteger("max_cache_len"))
      config.maxCacheLen = *v;
    if (auto v = obj->getInteger("attention_mask_len"))
      config.attentionMaskLen = *v;
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
  // Release decode buffers
  if (tokenView_) iree_hal_buffer_view_release(tokenView_);
  if (tokenBuffer_) iree_hal_buffer_release(tokenBuffer_);
  if (positionView_) iree_hal_buffer_view_release(positionView_);
  if (positionBuffer_) iree_hal_buffer_release(positionBuffer_);
  if (maskView_) iree_hal_buffer_view_release(maskView_);
  if (maskBuffer_) iree_hal_buffer_release(maskBuffer_);
  
  // Release cache view
  if (cacheView_) iree_hal_buffer_view_release(cacheView_);

  // Release IREE resources
  if (context_) iree_vm_context_release(context_);
  if (llmModule_) iree_vm_module_release(llmModule_);
  if (paramsModule_) iree_vm_module_release(paramsModule_);
  if (halModule_) iree_vm_module_release(halModule_);
  if (device_) iree_hal_device_release(device_);
  if (instance_) iree_vm_instance_release(instance_);
  
  tokenView_ = nullptr;
  tokenBuffer_ = nullptr;
  positionView_ = nullptr;
  positionBuffer_ = nullptr;
  maskView_ = nullptr;
  maskBuffer_ = nullptr;
  cacheView_ = nullptr;
  context_ = nullptr;
  llmModule_ = nullptr;
  paramsModule_ = nullptr;
  halModule_ = nullptr;
  device_ = nullptr;
  instance_ = nullptr;
  initialized_ = false;
}

llvm::Error IREEBackend::initialize() {
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Initializing\n");

  // Auto-detect config path from vmfb path
  std::string configPath = config_.configPath;
  if (configPath.empty() && !config_.vmfbPath.empty()) {
    llvm::SmallString<256> vmfbDir(config_.vmfbPath);
    llvm::sys::path::remove_filename(vmfbDir);
    llvm::sys::path::append(vmfbDir, "config.json");
    if (llvm::sys::fs::exists(vmfbDir)) {
      configPath = std::string(vmfbDir);
    }
  }

  // Load model config
  if (!configPath.empty()) {
    auto configOrErr = LLMModelConfig::loadFromFile(configPath);
    if (!configOrErr) {
      return configOrErr.takeError();
    }
    modelConfig_ = *configOrErr;
    LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Config loaded - layers=" 
                            << modelConfig_.numLayers 
                            << ", kv_heads=" << modelConfig_.numKVHeads
                            << ", head_dim=" << modelConfig_.headDim << "\n");
  }

  // Initialize tokenizer
  auto tokenizerOrErr = createTokenizer(config_.tokenizerPath);
  if (!tokenizerOrErr) {
    return tokenizerOrErr.takeError();
  }
  tokenizer_ = std::move(*tokenizerOrErr);
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Tokenizer loaded, vocab=" 
                          << tokenizer_->vocabSize() << "\n");

  // Create VM instance
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, hostAllocator_, &instance_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create VM instance: %s", msg.c_str());
  }

  // Register HAL types
  status = iree_hal_module_register_all_types(instance_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to register HAL types: %s", msg.c_str());
  }

  // Register drivers
  iree_hal_local_task_driver_module_register(iree_hal_driver_registry_default());
#ifdef IREE_HAVE_HAL_HIP_DRIVER
  iree_hal_hip_driver_module_register(iree_hal_driver_registry_default());
#endif

  // Create device
  iree_string_view_t deviceUri = iree_make_cstring_view(config_.deviceUri.c_str());
  status = iree_hal_create_device(iree_hal_driver_registry_default(),
                                   deviceUri, hostAllocator_, &device_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create device '%s': %s",
                                   config_.deviceUri.c_str(), msg.c_str());
  }
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Device created: " 
                          << config_.deviceUri << "\n");

  // Create HAL module
  iree_hal_module_debug_sink_t debugSink = {};
  status = iree_hal_module_create(
      instance_, iree_hal_module_device_policy_default(),
      /*device_count=*/1, &device_, IREE_HAL_MODULE_FLAG_NONE, debugSink,
      hostAllocator_, &halModule_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create HAL module: %s", msg.c_str());
  }

  // Load parameters from IRPA file
  iree_string_view_t irpaPath = iree_make_cstring_view(config_.irpaPath.c_str());
  iree_io_file_handle_t *fileHandle = nullptr;
  status = iree_io_file_handle_open(IREE_IO_FILE_MODE_READ, irpaPath,
                                     hostAllocator_, &fileHandle);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to open IRPA file: %s", msg.c_str());
  }

  iree_io_parameter_index_t *paramIndex = nullptr;
  status = iree_io_parameter_index_create(hostAllocator_, &paramIndex);
  if (!iree_status_is_ok(status)) {
    iree_io_file_handle_release(fileHandle);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create parameter index: %s", msg.c_str());
  }

  status = iree_io_parse_file_index(irpaPath, fileHandle, paramIndex,
                                    hostAllocator_);
  iree_io_file_handle_release(fileHandle);
  if (!iree_status_is_ok(status)) {
    iree_io_parameter_index_release(paramIndex);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to parse IRPA file: %s", msg.c_str());
  }

  // Create parameter provider
  iree_io_parameter_provider_t *paramProvider = nullptr;
  status = iree_io_parameter_index_provider_create(
      iree_make_cstring_view("model"), paramIndex,
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      hostAllocator_, &paramProvider);
  iree_io_parameter_index_release(paramIndex);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create parameter provider: %s", msg.c_str());
  }

  // Create parameters module
  status = iree_io_parameters_module_create(
      instance_, /*provider_count=*/1, &paramProvider, hostAllocator_, &paramsModule_);
  iree_io_parameter_provider_release(paramProvider);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create parameters module: %s", msg.c_str());
  }

  // Load VMFB
  auto vmfbOrErr = llvm::MemoryBuffer::getFile(config_.vmfbPath);
  if (!vmfbOrErr) {
    return llvm::createStringError(vmfbOrErr.getError(),
                                   "Failed to load VMFB: %s",
                                   config_.vmfbPath.c_str());
  }
  vmfbBuffer_ = std::move(*vmfbOrErr);

  iree_const_byte_span_t vmfbSpan = {
      reinterpret_cast<const uint8_t *>(vmfbBuffer_->getBufferStart()),
      vmfbBuffer_->getBufferSize()};
  status = iree_vm_bytecode_module_create(
      instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE, vmfbSpan,
      iree_allocator_null(), hostAllocator_, &llmModule_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create bytecode module: %s", msg.c_str());
  }

  // Create context with all modules
  std::vector<iree_vm_module_t *> modules = {paramsModule_, halModule_, llmModule_};
  status = iree_vm_context_create_with_modules(
      instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
      hostAllocator_, &context_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create context: %s", msg.c_str());
  }

  // Resolve functions
  status = iree_vm_context_resolve_function(
      context_, iree_make_cstring_view("module.prefill"), &prefillFn_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to find prefill function: %s", msg.c_str());
  }

  status = iree_vm_context_resolve_function(
      context_, iree_make_cstring_view("module.decode_step"), &decodeFn_);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to find decode_step function: %s", msg.c_str());
  }

  // Pre-allocate decode buffers
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device_);
  iree_hal_buffer_params_t bufParams = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
  };

  // Token buffer [1, 1]
  status = iree_hal_allocator_allocate_buffer(allocator, bufParams,
                                               sizeof(int64_t), &tokenBuffer_);
  if (iree_status_is_ok(status)) {
    const iree_hal_dim_t dims[] = {1, 1};
    status = iree_hal_buffer_view_create(tokenBuffer_, 2, dims,
                                          IREE_HAL_ELEMENT_TYPE_INT_64,
                                          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                          hostAllocator_, &tokenView_);
  }

  // Position buffer [1, 1]
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(allocator, bufParams,
                                                 sizeof(int64_t), &positionBuffer_);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_dim_t dims[] = {1, 1};
    status = iree_hal_buffer_view_create(positionBuffer_, 2, dims,
                                          IREE_HAL_ELEMENT_TYPE_INT_64,
                                          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                          hostAllocator_, &positionView_);
  }

  // Valid length buffer [1, 1] (used as valid_len for inplace cache model)
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(allocator, bufParams,
                                                 sizeof(int64_t), &maskBuffer_);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_dim_t dims[] = {1, 1};
    status = iree_hal_buffer_view_create(maskBuffer_, 2, dims,
                                          IREE_HAL_ELEMENT_TYPE_INT_64,
                                          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                          hostAllocator_, &maskView_);
  }

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to allocate decode buffers: %s", msg.c_str());
  }

  initialized_ = true;
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Initialization complete\n");
  return llvm::Error::success();
}

bool IREEBackend::isAvailable() const { return initialized_; }

int64_t IREEBackend::argmaxFp16(const uint16_t *logits, int size) {
  int64_t maxIdx = 0;
  uint16_t maxVal = 0;
  for (int i = 0; i < size; ++i) {
    uint16_t val = logits[i];
    uint16_t sign = val >> 15;
    uint16_t mag = val & 0x7FFF;
    uint16_t cmp = sign ? (0x8000 - mag) : (0x8000 + mag);
    if (cmp > maxVal) {
      maxVal = cmp;
      maxIdx = i;
    }
  }
  return maxIdx;
}

llvm::Expected<int64_t>
IREEBackend::runPrefill(llvm::ArrayRef<int64_t> tokens, int seqLen) {
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend::runPrefill: seqLen=" << seqLen << "\n");
  
  iree_status_t status = iree_ok_status();
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device_);

  int prefillLen = modelConfig_.prefillLen;
  
  // Pad tokens to prefillLen
  std::vector<int64_t> paddedTokens(prefillLen, 0);
  for (int i = 0; i < std::min(seqLen, prefillLen); ++i) {
    paddedTokens[i] = tokens[i];
  }

  // Create attention mask: 1 for valid, 0 for padding
  std::vector<int64_t> attentionMask(prefillLen, 0);
  for (int i = 0; i < std::min(seqLen, prefillLen); ++i) {
    attentionMask[i] = 1;
  }

  // Allocate tokens buffer
  iree_hal_buffer_view_t *tokensView = nullptr;
  const iree_hal_dim_t tokensDims[] = {1, static_cast<iree_hal_dim_t>(prefillLen)};
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device_, allocator, 2, tokensDims, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      },
      iree_make_const_byte_span(paddedTokens.data(), prefillLen * sizeof(int64_t)),
      &tokensView);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create tokens buffer: %s", msg.c_str());
  }

  // Allocate attention mask buffer
  iree_hal_buffer_view_t *maskView = nullptr;
  status = iree_hal_buffer_view_allocate_buffer_copy(
      device_, allocator, 2, tokensDims, IREE_HAL_ELEMENT_TYPE_INT_64,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      },
      iree_make_const_byte_span(attentionMask.data(), prefillLen * sizeof(int64_t)),
      &maskView);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(tokensView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create mask buffer: %s", msg.c_str());
  }

  // Build input list
  iree_vm_list_t *inputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                               hostAllocator_, &inputs);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(tokensView);
    iree_hal_buffer_view_release(maskView);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create input list: %s", msg.c_str());
  }

  iree_vm_ref_t tokensRef = iree_hal_buffer_view_move_ref(tokensView);
  iree_vm_ref_t maskRef = iree_hal_buffer_view_move_ref(maskView);
  iree_vm_list_push_ref_move(inputs, &tokensRef);
  iree_vm_list_push_ref_move(inputs, &maskRef);

  // Create output list
  iree_vm_list_t *outputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 3,
                               hostAllocator_, &outputs);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(inputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create output list: %s", msg.c_str());
  }

  // Invoke prefill
  status = iree_vm_invoke(context_, prefillFn_, IREE_VM_INVOCATION_FLAG_NONE,
                          nullptr, inputs, outputs, hostAllocator_);
  iree_vm_list_release(inputs);

  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to invoke prefill: %s", msg.c_str());
  }

  // Extract outputs: logits, cache (combined K/V)
  iree_vm_ref_t logitsRef = iree_vm_ref_null();
  iree_vm_list_get_ref_assign(outputs, 0, &logitsRef);
  iree_hal_buffer_view_t *logitsView = iree_hal_buffer_view_deref(logitsRef);

  iree_vm_ref_t cacheRef = iree_vm_ref_null();
  iree_vm_list_get_ref_assign(outputs, 1, &cacheRef);
  
  // Release old cache view and retain new one
  if (cacheView_) iree_hal_buffer_view_release(cacheView_);
  cacheView_ = iree_hal_buffer_view_deref(cacheRef);
  if (cacheView_) iree_hal_buffer_view_retain(cacheView_);

  // Find argmax at position seqLen-1
  iree_hal_buffer_t *logitsBuffer = iree_hal_buffer_view_buffer(logitsView);
  int vocabSize = modelConfig_.vocabSize;
  size_t rowOffset = (seqLen - 1) * vocabSize * sizeof(uint16_t);
  
  std::vector<uint16_t> logitsRow(vocabSize);
  status = iree_hal_device_transfer_d2h(
      device_, logitsBuffer, rowOffset,
      logitsRow.data(), vocabSize * sizeof(uint16_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());

  iree_vm_list_release(outputs);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to read logits: %s", msg.c_str());
  }

  int64_t nextToken = argmaxFp16(logitsRow.data(), vocabSize);
  currentPosition_ = seqLen;
  
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend::runPrefill: nextToken=" << nextToken << "\n");
  return nextToken;
}

llvm::Expected<int64_t>
IREEBackend::runDecode(int64_t token, int position) {
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend::runDecode: token=" << token 
                          << ", pos=" << position << "\n");

  if (!cacheView_) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Cache not initialized");
  }

  iree_status_t status = iree_ok_status();

  // Update token buffer [1,1]
  int64_t tokenVal = token;
  status = iree_hal_device_transfer_h2d(device_, &tokenVal, tokenBuffer_,
                                         0, sizeof(int64_t),
                                         IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                         iree_infinite_timeout());

  // Update position buffer [1,1]
  int64_t posVal = static_cast<int64_t>(position);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(device_, &posVal, positionBuffer_,
                                           0, sizeof(int64_t),
                                           IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                           iree_infinite_timeout());
  }

  // Update valid_len buffer [1,1] - number of valid positions (position + 1)
  int64_t validLen = static_cast<int64_t>(position + 1);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(device_, &validLen, maskBuffer_,
                                           0, sizeof(int64_t),
                                           IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                           iree_infinite_timeout());
  }

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to update inputs: %s", msg.c_str());
  }

  // Build input list: token, position_id, cache, valid_len
  iree_vm_list_t *inputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 4,
                               hostAllocator_, &inputs);
  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create input list: %s", msg.c_str());
  }

  // Retain views for reuse
  iree_hal_buffer_view_retain(tokenView_);
  iree_hal_buffer_view_retain(positionView_);
  iree_hal_buffer_view_retain(cacheView_);
  iree_hal_buffer_view_retain(maskView_);

  iree_vm_ref_t tokenRef = iree_hal_buffer_view_move_ref(tokenView_);
  iree_vm_ref_t posRef = iree_hal_buffer_view_move_ref(positionView_);
  iree_vm_ref_t cacheRef = iree_hal_buffer_view_move_ref(cacheView_);
  iree_vm_ref_t validLenRef = iree_hal_buffer_view_move_ref(maskView_);

  iree_vm_list_push_ref_move(inputs, &tokenRef);     // arg0: token [1,1]
  iree_vm_list_push_ref_move(inputs, &posRef);       // arg1: position_id [1,1]
  iree_vm_list_push_ref_move(inputs, &cacheRef);     // arg2: cache [cache_len,layers,2,heads,dim]
  iree_vm_list_push_ref_move(inputs, &validLenRef);  // arg3: valid_len [1,1]

  // Create output list
  iree_vm_list_t *outputs = nullptr;
  status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 2,
                               hostAllocator_, &outputs);
  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(inputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to create output list: %s", msg.c_str());
  }

  // Invoke decode_step
  status = iree_vm_invoke(context_, decodeFn_, IREE_VM_INVOCATION_FLAG_NONE,
                          nullptr, inputs, outputs, hostAllocator_);
  iree_vm_list_release(inputs);

  if (!iree_status_is_ok(status)) {
    iree_vm_list_release(outputs);
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to invoke decode_step: %s", msg.c_str());
  }

  // Extract logits and updated cache
  iree_vm_ref_t logitsRef = iree_vm_ref_null();
  iree_vm_list_get_ref_assign(outputs, 0, &logitsRef);
  iree_hal_buffer_view_t *logitsView = iree_hal_buffer_view_deref(logitsRef);

  // Update cache view with the returned cache (in-place updates are in the same buffer)
  iree_vm_ref_t newCacheRef = iree_vm_ref_null();
  iree_vm_list_get_ref_assign(outputs, 1, &newCacheRef);
  if (cacheView_) iree_hal_buffer_view_release(cacheView_);
  cacheView_ = iree_hal_buffer_view_deref(newCacheRef);
  if (cacheView_) iree_hal_buffer_view_retain(cacheView_);

  // Find argmax
  int vocabSize = modelConfig_.vocabSize;
  std::vector<uint16_t> logitsHost(vocabSize);
  iree_hal_buffer_t *logitsBuffer = iree_hal_buffer_view_buffer(logitsView);
  status = iree_hal_device_transfer_d2h(
      device_, logitsBuffer, 0, logitsHost.data(), vocabSize * sizeof(uint16_t),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());

  iree_vm_list_release(outputs);

  if (!iree_status_is_ok(status)) {
    auto msg = formatStatus(status);
    iree_status_free(status);
    return llvm::createStringError(std::errc::io_error,
                                   "Failed to read logits: %s", msg.c_str());
  }

  int64_t nextToken = argmaxFp16(logitsHost.data(), vocabSize);
  currentPosition_ = position + 1;
  
  return nextToken;
}

llvm::Expected<std::vector<int64_t>>
IREEBackend::generateTokens(llvm::ArrayRef<int64_t> promptTokens, int maxNewTokens) {
  int seqLen = promptTokens.size();
  int maxCacheLen = modelConfig_.maxCacheLen;

  // Prefill
  auto prefillStart = std::chrono::high_resolution_clock::now();
  auto firstTokenOrErr = runPrefill(promptTokens, seqLen);
  if (!firstTokenOrErr) {
    return firstTokenOrErr.takeError();
  }
  auto prefillEnd = std::chrono::high_resolution_clock::now();
  auto prefillMs = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();
  
  llvm::errs() << "IREEBackend: Prefill " << seqLen << " tokens in " 
               << static_cast<int>(prefillMs) << " ms\n";

  std::vector<int64_t> generatedTokens;
  generatedTokens.push_back(*firstTokenOrErr);
  int position = seqLen;

  // Decode loop
  auto decodeStart = std::chrono::high_resolution_clock::now();
  
  for (int i = 1; i < maxNewTokens && position < maxCacheLen - 1; ++i) {
    auto nextTokenOrErr = runDecode(generatedTokens.back(), position);
    if (!nextTokenOrErr) {
      return nextTokenOrErr.takeError();
    }

    int64_t nextToken = *nextTokenOrErr;
    generatedTokens.push_back(nextToken);
    position++;

    // Check for EOS (token ID 128001 for Llama 3, or configurable)
    // TODO: Get EOS token ID from tokenizer or config
    if (nextToken == 128001 || nextToken == 128009) {
      break;
    }
  }

  auto decodeEnd = std::chrono::high_resolution_clock::now();
  auto decodeMs = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
  int numDecodes = generatedTokens.size() - 1;
  double msPerToken = numDecodes > 0 ? decodeMs / numDecodes : 0;
  double tokPerSec = numDecodes > 0 ? numDecodes * 1000.0 / decodeMs : 0;
  
  llvm::errs() << "IREEBackend: Generated " << generatedTokens.size() 
               << " tokens in " << static_cast<int>(decodeMs) << " ms ("
               << llvm::format("%.1f", msPerToken) << " ms/tok, "
               << llvm::format("%.1f", tokPerSec) << " tok/s)\n";

  return generatedTokens;
}

llvm::Expected<GenerationResult>
IREEBackend::generate(llvm::StringRef prompt, const GenerationConfig &config) {
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Generating for prompt (" 
                          << prompt.size() << " chars)\n");

  auto startTime = std::chrono::steady_clock::now();

  // Tokenize
  std::vector<int64_t> tokens = tokenizer_->encode(prompt.str());
  LLVM_DEBUG(llvm::dbgs() << "IREEBackend: Tokenized to " << tokens.size() << " tokens\n");

  if (tokens.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Failed to tokenize prompt");
  }

  // Generate
  auto tokensOrErr = generateTokens(tokens, config.maxTokens);
  if (!tokensOrErr) {
    return tokensOrErr.takeError();
  }

  // Decode generated tokens
  std::string generatedText = tokenizer_->decode(*tokensOrErr);

  auto endTime = std::chrono::steady_clock::now();

  GenerationResult result;
  result.content = generatedText;
  result.promptTokens = tokens.size();
  result.completionTokens = tokensOrErr->size();
  result.latencyMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
  return result;
}

std::unique_ptr<LLMBackend>
createIREEBackend(const IREEBackendConfig &config) {
  auto backendOrErr = IREEBackend::create(config);
  if (!backendOrErr) {
    llvm::errs() << "Failed to create IREE backend: "
                 << toString(backendOrErr.takeError()) << "\n";
    return nullptr;
  }
  return std::move(*backendOrErr);
}

} // namespace mlir::iree_compiler::LLMAssist
