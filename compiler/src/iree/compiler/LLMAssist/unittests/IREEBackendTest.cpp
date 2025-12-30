// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Backend/IREEBackend.h"

#include "gtest/gtest.h"
#include "llvm/Support/FileSystem.h"

namespace mlir::iree_compiler::LLMAssist {
namespace {

class IREEBackendTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(IREEBackendTest, ConfigDefaults) {
  IREEBackendConfig config;
  EXPECT_EQ(config.deviceUri, "local-task");
  EXPECT_EQ(config.maxNewTokens, 512);
  EXPECT_EQ(config.maxSeqLen, 8192);
}

TEST_F(IREEBackendTest, CreateWithMissingVMFBFails) {
  IREEBackendConfig config;
  config.vmfbPath = "/nonexistent/path/to/model.vmfb";
  config.tokenizerPath = "/nonexistent/tokenizer.model";

  auto backendOrErr = IREEBackend::create(config);
  EXPECT_FALSE(static_cast<bool>(backendOrErr));

  // Consume the error
  llvm::consumeError(backendOrErr.takeError());
}

TEST_F(IREEBackendTest, ModelConfigDefaults) {
  LLMModelConfig config;
  EXPECT_EQ(config.numLayers, 32);
  EXPECT_EQ(config.numHeads, 32);
  EXPECT_EQ(config.numKVHeads, 8);
  EXPECT_EQ(config.headDim, 128);
  EXPECT_EQ(config.vocabSize, 32000);
  EXPECT_EQ(config.blockSeqStride, 16);
  EXPECT_EQ(config.modelType, "llama");
}

TEST_F(IREEBackendTest, ModelConfigLoadNonexistent) {
  auto configOrErr = LLMModelConfig::loadFromFile("/nonexistent/config.json");
  EXPECT_FALSE(static_cast<bool>(configOrErr));
  llvm::consumeError(configOrErr.takeError());
}

// Test with a valid JSON config file (if available)
class IREEBackendIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check if test files are available
    // These paths can be configured via environment variables
    const char *modelDir = std::getenv("LLM_ASSIST_TEST_MODEL_DIR");
    if (modelDir) {
      testModelDir_ = modelDir;
      hasTestFiles_ = true;
    }
  }

  std::string testModelDir_;
  bool hasTestFiles_ = false;
};

TEST_F(IREEBackendIntegrationTest, CreateWithValidConfig) {
  if (!hasTestFiles_) {
    GTEST_SKIP() << "Set LLM_ASSIST_TEST_MODEL_DIR to run integration tests";
  }

  IREEBackendConfig config;
  config.vmfbPath = testModelDir_ + "/model_gfx950.vmfb";
  config.irpaPath = testModelDir_ + "/model.irpa";
  config.tokenizerPath = testModelDir_ + "/tokenizer.model";
  config.configPath = testModelDir_ + "/config.json";
  config.deviceUri = "hip";

  auto backendOrErr = IREEBackend::create(config);
  if (!backendOrErr) {
    // If creation fails due to missing GPU, skip
    std::string errMsg = llvm::toString(backendOrErr.takeError());
    if (errMsg.find("hip") != std::string::npos ||
        errMsg.find("GPU") != std::string::npos) {
      GTEST_SKIP() << "GPU not available: " << errMsg;
    }
    FAIL() << "Unexpected error: " << errMsg;
  }

  auto &backend = *backendOrErr;
  EXPECT_TRUE(backend->isAvailable());
  EXPECT_EQ(backend->getName(), "IREE");
}

} // namespace
} // namespace mlir::iree_compiler::LLMAssist

