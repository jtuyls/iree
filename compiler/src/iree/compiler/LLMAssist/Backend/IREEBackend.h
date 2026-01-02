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
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// Configuration for the IREE LLM backend.
struct IREEBackendConfig {
  /// Path to the compiled LLM module (.vmfb file).
  std::string vmfbPath;

  /// Path to the model parameters (.irpa file).
  std::string irpaPath;

  /// Path to the tokenizer model.
  std::string tokenizerPath;

  /// Path to the model configuration JSON (optional, auto-detected if empty).
  std::string configPath;

  /// IREE device URI (e.g., "local-task", "hip", "cuda://0").
  std::string deviceUri = "local-task";

  /// Maximum number of tokens to generate.
  int maxNewTokens = 512;
};

/// LLM model configuration loaded from JSON.
struct LLMModelConfig {
  // Model architecture
  int numLayers = 32;
  int numKVHeads = 8;
  int headDim = 128;
  int vocabSize = 128256;
  
  // Cache configuration
  int prefillLen = 512;         // Fixed prefill sequence length
  int maxCacheLen = 512;        // Maximum KV cache length
  int attentionMaskLen = 513;   // prefillLen + 1 for decode

  /// Load configuration from a JSON file.
  static llvm::Expected<LLMModelConfig> loadFromFile(llvm::StringRef path);
};

/// IREE-native LLM backend for in-process inference.
///
/// This backend loads a pre-compiled LLM VMFB module with contiguous KV cache
/// layout and runs inference directly within the compiler process.
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
  generate(llvm::StringRef prompt, const GenerationConfig &config) override;

  /// Get the backend name.
  llvm::StringRef getName() const override { return "IREE"; }

  /// Get the loaded model configuration.
  const LLMModelConfig &getModelConfig() const { return modelConfig_; }

private:
  IREEBackend(const IREEBackendConfig &config);

  llvm::Error initialize();
  void cleanup();

  /// Internal token generation loop.
  llvm::Expected<std::vector<int64_t>>
  generateTokens(llvm::ArrayRef<int64_t> promptTokens, int maxNewTokens);

  /// Run the prefill phase (process prompt, initialize cache).
  llvm::Expected<int64_t> runPrefill(llvm::ArrayRef<int64_t> tokens, int seqLen);

  /// Run a single decode step.
  llvm::Expected<int64_t> runDecode(int64_t token, int position);

  /// Find argmax of fp16 logits buffer.
  int64_t argmaxFp16(const uint16_t *logits, int size);

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

  // KV Cache buffers (contiguous layout: [max_cache_len, layers, heads, dim])
  iree_hal_buffer_view_t *cacheKView_ = nullptr;
  iree_hal_buffer_view_t *cacheVView_ = nullptr;

  // Pre-allocated decode input buffers (avoid allocation per step)
  iree_hal_buffer_t *tokenBuffer_ = nullptr;
  iree_hal_buffer_view_t *tokenView_ = nullptr;
  iree_hal_buffer_t *positionBuffer_ = nullptr;
  iree_hal_buffer_view_t *positionView_ = nullptr;
  iree_hal_buffer_t *maskBuffer_ = nullptr;
  iree_hal_buffer_view_t *maskView_ = nullptr;

  // Host allocator
  iree_allocator_t hostAllocator_;

  // Current sequence position
  int currentPosition_ = 0;
};

/// Factory function to create an IREE backend.
std::unique_ptr<LLMBackend> createIREEBackend(const IREEBackendConfig &config);

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_BACKEND_IREEBACKEND_H_
