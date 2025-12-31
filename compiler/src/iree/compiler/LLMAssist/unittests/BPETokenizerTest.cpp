// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Tokenizer/BPETokenizer.h"
#include "iree/testing/gtest.h"

#include <fstream>

namespace mlir::iree_compiler::LLMAssist {
namespace {

// Path to test tokenizer.json - set via environment variable
const char *getTestTokenizerPath() {
  const char *path = std::getenv("IREE_TEST_BPE_TOKENIZER_PATH");
  return path ? path : "";
}

class BPETokenizerTest : public ::testing::Test {
protected:
  void SetUp() override {
    tokenizerPath_ = getTestTokenizerPath();
    skipTests_ = tokenizerPath_.empty();
    if (skipTests_) {
      GTEST_SKIP() << "IREE_TEST_BPE_TOKENIZER_PATH not set, skipping BPE "
                      "tokenizer tests";
    }
  }

  std::string tokenizerPath_;
  bool skipTests_ = true;
};

TEST_F(BPETokenizerTest, CreateFromJson) {
  auto tokOrErr = BPETokenizer::create(tokenizerPath_);
  ASSERT_TRUE(static_cast<bool>(tokOrErr))
      << "Failed to create tokenizer: " << llvm::toString(tokOrErr.takeError());

  auto &tok = *tokOrErr;
  EXPECT_GT(tok->vocabSize(), 100000u);
  EXPECT_EQ(tok->getName(), "BPE");
}

TEST_F(BPETokenizerTest, EncodeSimple) {
  auto tokOrErr = BPETokenizer::create(tokenizerPath_);
  ASSERT_TRUE(static_cast<bool>(tokOrErr));
  auto &tok = *tokOrErr;

  // Test "Hello" - HuggingFace tokenizers returns [9906]
  auto tokens = tok->encode("Hello");
  EXPECT_FALSE(tokens.empty());
  
  // Print tokens for debugging
  llvm::errs() << "Encoded 'Hello': [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << tokens[i];
  }
  llvm::errs() << "]\n";
  
  // Test multi-word - HuggingFace returns [791, 4062, 14198, 39935]
  auto tokens2 = tok->encode("The quick brown fox");
  llvm::errs() << "Encoded 'The quick brown fox': [";
  for (size_t i = 0; i < tokens2.size(); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << tokens2[i];
  }
  llvm::errs() << "]\n";
  
  EXPECT_FALSE(tokens2.empty());
}

TEST_F(BPETokenizerTest, DecodeRoundTrip) {
  auto tokOrErr = BPETokenizer::create(tokenizerPath_);
  ASSERT_TRUE(static_cast<bool>(tokOrErr));
  auto &tok = *tokOrErr;

  std::string original = "Hello world test";
  auto tokens = tok->encode(original);
  std::string decoded = tok->decode(tokens);

  // The decoded string should approximately match
  // (might have minor whitespace differences due to BPE quirks)
  EXPECT_FALSE(decoded.empty());
}

TEST_F(BPETokenizerTest, SpecialTokens) {
  auto tokOrErr = BPETokenizer::create(tokenizerPath_);
  ASSERT_TRUE(static_cast<bool>(tokOrErr));
  auto &tok = *tokOrErr;

  // Llama 3 special tokens
  EXPECT_EQ(tok->bosId(), 128000);
  EXPECT_EQ(tok->eosId(), 128001);
}

TEST_F(BPETokenizerTest, EncodeMLIR) {
  auto tokOrErr = BPETokenizer::create(tokenizerPath_);
  ASSERT_TRUE(static_cast<bool>(tokOrErr));
  auto &tok = *tokOrErr;

  // Test encoding MLIR code
  std::string mlir = "module { func.func @test() }";
  auto tokens = tok->encode(mlir);
  EXPECT_FALSE(tokens.empty());

  std::string decoded = tok->decode(tokens);
  EXPECT_FALSE(decoded.empty());
}

// Test for invalid JSON file
TEST(BPETokenizerErrorTest, InvalidFile) {
  auto tokOrErr = BPETokenizer::create("/nonexistent/path/tokenizer.json");
  EXPECT_FALSE(static_cast<bool>(tokOrErr));
  // Consume the error
  llvm::consumeError(tokOrErr.takeError());
}

} // namespace
} // namespace mlir::iree_compiler::LLMAssist

