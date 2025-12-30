// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_BACKEND_LLMBACKEND_H_
#define IREE_COMPILER_LLMASSIST_BACKEND_LLMBACKEND_H_

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::iree_compiler::LLMAssist {

/// Configuration for LLM generation.
struct GenerationConfig {
  /// Temperature for sampling (0.0 = deterministic).
  float temperature = 0.0f;
  /// Maximum number of tokens to generate.
  int maxTokens = 4096;
  /// Random seed for reproducibility.
  std::optional<int> seed;
  /// Model name to use.
  std::string model = "qwen2.5-coder:7b";
};

/// Result of an LLM generation request.
struct GenerationResult {
  /// Generated content.
  std::string content;
  /// Number of prompt tokens.
  int promptTokens = 0;
  /// Number of completion tokens.
  int completionTokens = 0;
  /// Generation latency in milliseconds.
  float latencyMs = 0.0f;
};

/// Abstract interface for LLM backends.
///
/// This interface allows different LLM implementations (Ollama, IREE native,
/// etc.) to be used interchangeably by the LLM-assisted passes.
class LLMBackend {
public:
  virtual ~LLMBackend() = default;

  /// Check if the backend is available and ready to serve requests.
  virtual bool isAvailable() const = 0;

  /// Generate a completion for the given prompt.
  ///
  /// Returns the generation result on success, or an error on failure.
  virtual llvm::Expected<GenerationResult>
  generate(llvm::StringRef prompt, const GenerationConfig &config) = 0;

  /// Get the backend name for logging.
  virtual llvm::StringRef getName() const = 0;
};

/// Create an Ollama HTTP backend.
///
/// \param endpoint The Ollama API endpoint (e.g., "http://localhost:11434").
std::unique_ptr<LLMBackend> createOllamaBackend(llvm::StringRef endpoint);

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_BACKEND_LLMBACKEND_H_

