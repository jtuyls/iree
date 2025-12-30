// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_BACKEND_IREEBACKEND_H_
#define IREE_COMPILER_LLMASSIST_BACKEND_IREEBACKEND_H_

#include "iree/compiler/LLMAssist/Backend/LLMBackend.h"
#include "iree/compiler/LLMAssist/Tokenizer/Tokenizer.h"

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>

namespace mlir::iree_compiler::LLMAssist {

/// Configuration for the IREE LLM backend.
struct IREEBackendConfig {
  /// Path to the compiled LLM module (.vmfb file).
  std::string vmfbPath;

  /// Path to the model parameters (.irpa file).
  std::string irpaPath;

  /// Path to the tokenizer model (e.g., .model file for SentencePiece).
  std::string tokenizerPath;

  /// Path to the model configuration JSON.
  std::string configPath;

  /// IREE device URI (e.g., "local-task", "rocm://0", "cuda://0").
  std::string deviceUri = "local-task";

  /// Maximum number of tokens to generate.
  int maxNewTokens = 512;

  /// Maximum sequence length (prompt + generated).
  int maxSeqLen = 8192;
};

/// LLM model configuration loaded from JSON.
struct LLMModelConfig {
  int numLayers = 32;
  int numHeads = 32;
  int numKVHeads = 8;
  int headDim = 128;
  int vocabSize = 32000;
  int blockSeqStride = 16;
  std::string modelType = "llama";

  /// Load configuration from a JSON file.
  static llvm::Expected<LLMModelConfig> loadFromFile(llvm::StringRef path);
};

/// IREE-native LLM backend for in-process inference.
///
/// This backend loads a pre-compiled LLM VMFB module and runs inference
/// directly within the compiler process, without requiring external services.
class IREEBackend : public LLMBackend {
public:
  /// Create an IREE backend with the given configuration.
  static llvm::Expected<std::unique_ptr<IREEBackend>>
  create(const IREEBackendConfig &config);

  ~IREEBackend();

  // Non-copyable, non-movable
  IREEBackend(const IREEBackend &) = delete;
  IREEBackend &operator=(const IREEBackend &) = delete;

  /// Check if the backend is properly initialized and ready.
  bool isAvailable() const override;

  /// Generate text completion.
  llvm::Expected<GenerationResult>
  generate(llvm::StringRef prompt,
           const GenerationConfig &config) override;

  /// Get the backend name.
  llvm::StringRef getName() const override { return "IREE"; }

  /// Get the loaded model configuration.
  const LLMModelConfig &getModelConfig() const { return modelConfig_; }

private:
  IREEBackend(const IREEBackendConfig &config);

  llvm::Error initialize();

  /// Internal token generation loop.
  llvm::Expected<std::vector<int64_t>>
  generateTokens(llvm::ArrayRef<int64_t> promptTokens, int maxNewTokens);

  /// Run the prefill phase (process prompt).
  /// Returns the first predicted token.
  llvm::Expected<int64_t> runPrefill(llvm::ArrayRef<int64_t> tokens, int seqLen);

  /// Run a single decode step.
  /// Returns the next predicted token.
  llvm::Expected<int64_t> runDecode(int64_t token, int seqLen, int startPos);

  /// Initialize the KV-cache buffer.
  llvm::Error initializeCache();

  /// Clean up IREE resources.
  void cleanup();

  IREEBackendConfig config_;
  LLMModelConfig modelConfig_;
  bool initialized_ = false;

  // Tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // IREE runtime objects
  iree_vm_instance_t *instance_ = nullptr;
  iree_hal_device_t *device_ = nullptr;
  iree_vm_context_t *context_ = nullptr;
  iree_vm_module_t *halModule_ = nullptr;
  iree_vm_module_t *paramsModule_ = nullptr;
  iree_vm_module_t *llmModule_ = nullptr;

  // VMFB buffer (must stay alive for the module's lifetime)
  std::unique_ptr<llvm::MemoryBuffer> vmfbBuffer_;

  // Function handles
  iree_vm_function_t prefillFn_;
  iree_vm_function_t decodeFn_;

  // Flat KV-cache buffer (matches Python test approach)
  // Shape: [deviceBlockCount, pageSize] where pageSize = layers*2*heads*stride*dim
  iree_hal_buffer_t *cacheBuffer_ = nullptr;
  iree_hal_buffer_view_t *cacheBufferView_ = nullptr;
  int64_t deviceBlockCount_ = 64;
  int64_t pageSize_ = 0;  // Computed from model config
  int blockSeqStride_ = 32;

  // Pre-allocated decode input buffers and views (for performance)
  iree_hal_buffer_t *decodeTokenBuffer_ = nullptr;
  iree_hal_buffer_view_t *decodeTokenView_ = nullptr;
  iree_hal_buffer_t *decodeSeqLenBuffer_ = nullptr;
  iree_hal_buffer_view_t *decodeSeqLenView_ = nullptr;
  iree_hal_buffer_t *decodeStartPosBuffer_ = nullptr;
  iree_hal_buffer_view_t *decodeStartPosView_ = nullptr;
  iree_hal_buffer_t *decodePageTableBuffer_ = nullptr;
  int decodePageTableCapacity_ = 0;  // Current capacity in pages
  int currentNumPages_ = 0;  // Current page table size

  // Host allocator
  iree_allocator_t hostAllocator_;
};

/// Factory function to create an IREE backend.
std::unique_ptr<LLMBackend> createIREEBackend(const IREEBackendConfig &config);

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_BACKEND_IREEBACKEND_H_

