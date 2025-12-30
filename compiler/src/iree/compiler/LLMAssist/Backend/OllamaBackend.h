// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_BACKEND_OLLAMABACKEND_H_
#define IREE_COMPILER_LLMASSIST_BACKEND_OLLAMABACKEND_H_

#include "iree/compiler/LLMAssist/Backend/LLMBackend.h"

namespace mlir::iree_compiler::LLMAssist {

/// Ollama HTTP backend for LLM inference.
///
/// This backend communicates with an Ollama server via HTTP to perform
/// LLM inference. Ollama provides a simple API for running various open-source
/// LLMs locally.
///
/// See: https://ollama.ai/
class OllamaBackend : public LLMBackend {
public:
  /// Create an Ollama backend with the given endpoint.
  ///
  /// \param endpoint The Ollama API endpoint (e.g., "http://localhost:11434").
  explicit OllamaBackend(llvm::StringRef endpoint);

  ~OllamaBackend() override;

  /// Check if Ollama server is reachable.
  bool isAvailable() const override;

  /// Generate completion using Ollama API.
  llvm::Expected<GenerationResult>
  generate(llvm::StringRef prompt, const GenerationConfig &config) override;

  llvm::StringRef getName() const override { return "Ollama"; }

private:
  std::string endpoint_;
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_BACKEND_OLLAMABACKEND_H_

