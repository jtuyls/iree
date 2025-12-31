// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_TOKENIZER_BPETOKENIZER_H
#define IREE_COMPILER_LLMASSIST_TOKENIZER_BPETOKENIZER_H

#include "iree/compiler/LLMAssist/Tokenizer/Tokenizer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// BPE (Byte-Pair Encoding) tokenizer for Llama 3.x and similar models.
///
/// This implements the HuggingFace tokenizers BPE format, which is used by
/// Llama 3, GPT-4, and other modern LLMs. It parses the tokenizer.json format
/// and implements byte-level BPE encoding.
class BPETokenizer : public Tokenizer {
public:
  /// Create a BPE tokenizer from a HuggingFace tokenizer.json file.
  static llvm::Expected<std::unique_ptr<BPETokenizer>>
  create(llvm::StringRef tokenizerJsonPath);

  ~BPETokenizer() override = default;

  /// Encode text to token IDs.
  std::vector<int64_t> encode(llvm::StringRef text) const override;

  /// Decode token IDs to text.
  std::string decode(llvm::ArrayRef<int64_t> tokens) const override;

  /// Get vocabulary size.
  size_t vocabSize() const override { return vocab_.size(); }

  /// Get special token IDs.
  int64_t bosId() const override { return bosId_; }
  int64_t eosId() const override { return eosId_; }
  int64_t padId() const override { return padId_; }

  /// Decode a single token to text.
  std::string decodeToken(int64_t tokenId) const override;

  /// Get tokenizer name.
  llvm::StringRef getName() const override { return "BPE"; }

private:
  BPETokenizer() = default;

  /// Load tokenizer from JSON file.
  llvm::Error loadFromJson(llvm::StringRef path);

  /// Pre-tokenize text into chunks using the regex pattern.
  std::vector<std::string> preTokenize(llvm::StringRef text) const;

  /// Apply BPE to a single pre-tokenized chunk.
  std::vector<int64_t> bpeEncode(const std::string &chunk) const;

  /// Convert a UTF-8 string to byte tokens (using Ġ encoding for bytes).
  std::vector<std::string> bytesToTokens(const std::string &text) const;

  /// Get the rank of a merge (lower = higher priority).
  /// Returns -1 if the merge doesn't exist.
  int getMergeRank(const std::string &token1, const std::string &token2) const;

  /// Encode a segment of text (without special tokens).
  std::vector<int64_t> encodeSegment(const std::string &text, bool isFirst) const;

  // Vocabulary: token string -> ID
  std::unordered_map<std::string, int64_t> vocab_;

  // Reverse vocabulary: ID -> token string
  std::unordered_map<int64_t, std::string> idToToken_;

  // Merge rules: (token1 + " " + token2) -> rank (priority)
  std::unordered_map<std::string, int> mergeRanks_;

  // Byte-to-token mapping (for byte fallback)
  std::array<std::string, 256> byteToToken_;

  // Special token IDs
  int64_t bosId_ = 128000;
  int64_t eosId_ = 128001;
  int64_t padId_ = -1; // Llama 3 doesn't have a dedicated pad token

  // Pre-tokenization pattern (similar to GPT-4)
  std::regex preTokenizePattern_;
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_TOKENIZER_BPETOKENIZER_H

