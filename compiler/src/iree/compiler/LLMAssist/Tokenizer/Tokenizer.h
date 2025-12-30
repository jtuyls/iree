// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_TOKENIZER_TOKENIZER_H_
#define IREE_COMPILER_LLMASSIST_TOKENIZER_TOKENIZER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// Abstract interface for text tokenizers.
class Tokenizer {
public:
  virtual ~Tokenizer() = default;

  /// Encode text to token IDs.
  virtual std::vector<int64_t> encode(llvm::StringRef text) = 0;

  /// Decode token IDs back to text.
  virtual std::string decode(llvm::ArrayRef<int64_t> ids) = 0;

  /// Decode a single token ID to text.
  virtual std::string decodeToken(int64_t id) = 0;

  /// Get the vocabulary size.
  virtual size_t vocabSize() const = 0;

  /// Get special token IDs.
  virtual int64_t bosId() const = 0;  // Beginning of sequence
  virtual int64_t eosId() const = 0;  // End of sequence
  virtual int64_t padId() const = 0;  // Padding token

  /// Get the tokenizer name/type.
  virtual llvm::StringRef getName() const = 0;
};

/// Factory function to create a SentencePiece tokenizer.
/// Returns nullptr if SentencePiece is not available.
llvm::Expected<std::unique_ptr<Tokenizer>>
createSentencePieceTokenizer(llvm::StringRef modelPath);

/// Check if SentencePiece tokenizer support is available.
bool isSentencePieceAvailable();

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_TOKENIZER_TOKENIZER_H_


