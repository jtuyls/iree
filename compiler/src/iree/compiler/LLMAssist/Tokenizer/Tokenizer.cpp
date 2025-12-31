// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Tokenizer/Tokenizer.h"
#include "iree/compiler/LLMAssist/Tokenizer/BPETokenizer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace mlir::iree_compiler::LLMAssist {

#ifndef IREE_LLM_ASSIST_HAVE_SENTENCEPIECE

// Stub implementation when SentencePiece is not available

bool isSentencePieceAvailable() { return false; }

llvm::Expected<std::unique_ptr<Tokenizer>>
createSentencePieceTokenizer(llvm::StringRef modelPath) {
  return llvm::createStringError(
      std::errc::not_supported,
      "SentencePiece tokenizer is not available. "
      "Rebuild with IREE_LLM_ASSIST_ENABLE_SENTENCEPIECE=ON");
}

#endif // !IREE_LLM_ASSIST_HAVE_SENTENCEPIECE

llvm::Expected<std::unique_ptr<Tokenizer>>
createBPETokenizer(llvm::StringRef tokenizerJsonPath) {
  return BPETokenizer::create(tokenizerJsonPath);
}

llvm::Expected<std::unique_ptr<Tokenizer>>
createTokenizer(llvm::StringRef path) {
  // Determine tokenizer type based on file extension
  llvm::StringRef ext = llvm::sys::path::extension(path);

  if (ext == ".json") {
    // HuggingFace tokenizer.json format (BPE for Llama 3, GPT-4, etc.)
    return createBPETokenizer(path);
  } else if (ext == ".model") {
    // SentencePiece format (Llama 2, etc.)
    return createSentencePieceTokenizer(path);
  } else {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "Unknown tokenizer file format: %s. "
        "Expected .json (BPE) or .model (SentencePiece)",
        path.str().c_str());
  }
}

} // namespace mlir::iree_compiler::LLMAssist


