// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Tokenizer/Tokenizer.h"
#include "llvm/Support/Error.h"

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

} // namespace mlir::iree_compiler::LLMAssist


