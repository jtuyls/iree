// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Tokenizer/BPETokenizer.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

namespace mlir::iree_compiler::LLMAssist {

namespace {

// Byte-to-unicode mapping used by GPT-2/Llama 3 tokenizers.
// This maps bytes 0-255 to printable unicode characters.
// See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
std::array<std::string, 256> createByteToUnicode() {
  std::array<std::string, 256> result;

  // Printable ASCII characters map to themselves
  // Control characters and special bytes map to unicode range 256+
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
        (b >= 0xAE && b <= 0xFF)) {
      // Printable: map to self
      result[b] = std::string(1, static_cast<char>(b));
    } else {
      // Non-printable: map to unicode character 256 + n
      // These are encoded as UTF-8 multi-byte sequences
      int codepoint = 256 + n;
      ++n;
      // Encode codepoint as UTF-8
      if (codepoint < 0x80) {
        result[b] = std::string(1, static_cast<char>(codepoint));
      } else if (codepoint < 0x800) {
        result[b] = std::string{static_cast<char>(0xC0 | (codepoint >> 6)),
                                static_cast<char>(0x80 | (codepoint & 0x3F))};
      } else {
        result[b] = std::string{
            static_cast<char>(0xE0 | (codepoint >> 12)),
            static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)),
            static_cast<char>(0x80 | (codepoint & 0x3F))};
      }
    }
  }
  return result;
}

// Create reverse mapping from unicode character to byte
std::unordered_map<std::string, uint8_t> createUnicodeToByte() {
  auto byteToUnicode = createByteToUnicode();
  std::unordered_map<std::string, uint8_t> result;
  for (int b = 0; b < 256; ++b) {
    result[byteToUnicode[b]] = static_cast<uint8_t>(b);
  }
  return result;
}

} // namespace

llvm::Expected<std::unique_ptr<BPETokenizer>>
BPETokenizer::create(llvm::StringRef tokenizerJsonPath) {
  auto tokenizer = std::unique_ptr<BPETokenizer>(new BPETokenizer());

  if (auto err = tokenizer->loadFromJson(tokenizerJsonPath)) {
    return std::move(err);
  }

  return tokenizer;
}

llvm::Error BPETokenizer::loadFromJson(llvm::StringRef path) {
  // Read the file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    return llvm::createStringError(bufferOrErr.getError(),
                                   "Failed to open tokenizer file: %s",
                                   path.str().c_str());
  }

  // Parse JSON
  auto jsonOrErr = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!jsonOrErr) {
    return jsonOrErr.takeError();
  }

  auto *root = jsonOrErr->getAsObject();
  if (!root) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Invalid tokenizer JSON: not an object");
  }

  // Get model section
  auto *model = root->getObject("model");
  if (!model) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Invalid tokenizer JSON: missing 'model'");
  }

  // Check model type
  if (auto type = model->getString("type")) {
    if (*type != "BPE") {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Unsupported tokenizer type: %s (expected BPE)", type->str().c_str());
    }
  }

  // Load vocabulary
  auto *vocab = model->getObject("vocab");
  if (!vocab) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Invalid tokenizer JSON: missing 'vocab'");
  }

  for (const auto &entry : *vocab) {
    std::string token = entry.first.str();
    if (auto id = entry.second.getAsInteger()) {
      vocab_[token] = *id;
      idToToken_[*id] = token;
    }
  }

  llvm::errs() << "BPETokenizer: Loaded " << vocab_.size() << " vocab entries\n";

  // Load merge rules
  auto *merges = model->getArray("merges");
  if (merges) {
    int rank = 0;
    for (const auto &merge : *merges) {
      if (auto mergeStr = merge.getAsString()) {
        mergeRanks_[mergeStr->str()] = rank++;
      }
    }
    llvm::errs() << "BPETokenizer: Loaded " << mergeRanks_.size()
                 << " merge rules\n";
  }

  // Load added tokens (special tokens)
  auto *addedTokens = root->getArray("added_tokens");
  if (addedTokens) {
    for (const auto &token : *addedTokens) {
      if (auto *tokenObj = token.getAsObject()) {
        auto content = tokenObj->getString("content");
        auto id = tokenObj->getInteger("id");
        if (content && id) {
          vocab_[content->str()] = *id;
          idToToken_[*id] = content->str();

          // Check for special tokens
          if (*content == "<|begin_of_text|>") {
            bosId_ = *id;
          } else if (*content == "<|end_of_text|>") {
            eosId_ = *id;
          }
        }
      }
    }
  }

  // Initialize byte-to-token mapping
  byteToToken_ = createByteToUnicode();

  // Create pre-tokenization regex pattern
  // This is the GPT-4 pattern that splits on whitespace, punctuation, etc.
  // Note: C++ regex has limited unicode support, so we use a simplified pattern
  // IREE is compiled with -fno-exceptions, so we can't use try/catch for regex
  // errors. The pattern below is simple enough that it shouldn't fail.
  preTokenizePattern_ = std::regex(
      "('s|'t|'re|'ve|'m|'ll|'d)|"        // Contractions
      "[a-zA-Z]+|"                         // Words
      "[0-9]+|"                            // Numbers
      "[^a-zA-Z0-9\\s]+|"                  // Punctuation
      "\\s+",                              // Whitespace
      std::regex::ECMAScript | std::regex::optimize);

  return llvm::Error::success();
}

std::vector<std::string> BPETokenizer::preTokenize(llvm::StringRef text) const {
  std::vector<std::string> result;
  std::string str = text.str();

  // Use regex to split into chunks
  std::sregex_iterator it(str.begin(), str.end(), preTokenizePattern_);
  std::sregex_iterator end;

  for (; it != end; ++it) {
    std::string match = it->str();
    if (!match.empty()) {
      result.push_back(match);
    }
  }

  return result;
}

std::vector<std::string>
BPETokenizer::bytesToTokens(const std::string &text) const {
  std::vector<std::string> result;

  for (unsigned char c : text) {
    result.push_back(byteToToken_[c]);
  }

  return result;
}

int BPETokenizer::getMergeRank(const std::string &token1,
                               const std::string &token2) const {
  std::string key = token1 + " " + token2;
  auto it = mergeRanks_.find(key);
  if (it != mergeRanks_.end()) {
    return it->second;
  }
  return -1;
}

std::vector<int64_t> BPETokenizer::bpeEncode(const std::string &chunk) const {
  // Check if the whole chunk is in vocabulary (common for short tokens)
  auto directIt = vocab_.find(chunk);
  if (directIt != vocab_.end()) {
    return {directIt->second};
  }

  // Split into individual UTF-8 characters as initial tokens
  std::vector<std::string> tokens;
  size_t i = 0;
  while (i < chunk.size()) {
    // Determine UTF-8 character length
    unsigned char c = static_cast<unsigned char>(chunk[i]);
    size_t charLen = 1;
    if ((c & 0xE0) == 0xC0)
      charLen = 2; // 2-byte UTF-8
    else if ((c & 0xF0) == 0xE0)
      charLen = 3; // 3-byte UTF-8
    else if ((c & 0xF8) == 0xF0)
      charLen = 4; // 4-byte UTF-8

    if (i + charLen <= chunk.size()) {
      tokens.push_back(chunk.substr(i, charLen));
    }
    i += charLen;
  }

  // Iteratively apply BPE merges
  while (tokens.size() > 1) {
    // Find the pair with the lowest merge rank
    int bestRank = INT_MAX;
    size_t bestIdx = SIZE_MAX;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
      int rank = getMergeRank(tokens[i], tokens[i + 1]);
      if (rank >= 0 && rank < bestRank) {
        bestRank = rank;
        bestIdx = i;
      }
    }

    // No more merges possible
    if (bestIdx == SIZE_MAX) {
      break;
    }

    // Apply the merge
    std::string merged = tokens[bestIdx] + tokens[bestIdx + 1];
    tokens[bestIdx] = merged;
    tokens.erase(tokens.begin() + bestIdx + 1);
  }

  // Convert tokens to IDs
  std::vector<int64_t> result;
  for (const auto &token : tokens) {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
      result.push_back(it->second);
    } else {
      // Unknown token - try to find as individual characters
      llvm::errs() << "BPETokenizer: Unknown token after BPE: '" << token
                   << "' (len=" << token.size() << ")\n";
    }
  }

  return result;
}

std::vector<int64_t> BPETokenizer::encode(llvm::StringRef text) const {
  std::vector<int64_t> result;

  // Pre-tokenize into chunks
  std::vector<std::string> chunks = preTokenize(text);

  // Process chunks, handling spaces correctly
  // In GPT-style tokenizers, spaces are prepended to the following token as Ġ
  bool pendingSpace = false;
  bool isFirst = true;
  
  for (const auto &chunk : chunks) {
    // Check if this chunk is purely whitespace
    bool isWhitespace = true;
    for (char c : chunk) {
      if (!std::isspace(static_cast<unsigned char>(c))) {
        isWhitespace = false;
        break;
      }
    }
    
    if (isWhitespace) {
      // Mark that the next token should get a space prefix
      pendingSpace = true;
      continue;
    }
    
    // Build the token with appropriate space prefix
    std::string processedChunk;
    
    if (pendingSpace || (!isFirst && !chunk.empty())) {
      // Add Ġ prefix (U+0120 = 0xC4 0xA0 in UTF-8)
      processedChunk = "\xC4\xA0" + chunk;
    } else {
      processedChunk = chunk;
    }
    
    pendingSpace = false;
    isFirst = false;

    if (!processedChunk.empty()) {
      auto ids = bpeEncode(processedChunk);
      result.insert(result.end(), ids.begin(), ids.end());
    }
  }

  return result;
}

std::string BPETokenizer::decode(llvm::ArrayRef<int64_t> tokens) const {
  std::string result;

  // Create unicode-to-byte mapping
  static auto unicodeToByte = createUnicodeToByte();

  for (int64_t id : tokens) {
    auto it = idToToken_.find(id);
    if (it != idToToken_.end()) {
      const std::string &token = it->second;

      // Skip special tokens in decoded output
      if (token.find("<|") == 0) {
        continue;
      }

      // Convert token characters back to bytes
      std::string decoded;
      size_t i = 0;
      while (i < token.size()) {
        // Try to match multi-byte unicode sequences
        bool found = false;
        for (int len = 3; len >= 1 && !found; --len) {
          if (i + len <= token.size()) {
            std::string sub = token.substr(i, len);
            auto byteIt = unicodeToByte.find(sub);
            if (byteIt != unicodeToByte.end()) {
              decoded += static_cast<char>(byteIt->second);
              i += len;
              found = true;
            }
          }
        }
        if (!found) {
          // Couldn't decode, just copy the character
          decoded += token[i];
          ++i;
        }
      }

      result += decoded;
    }
  }

  return result;
}

std::string BPETokenizer::decodeToken(int64_t tokenId) const {
  return decode({tokenId});
}

} // namespace mlir::iree_compiler::LLMAssist

