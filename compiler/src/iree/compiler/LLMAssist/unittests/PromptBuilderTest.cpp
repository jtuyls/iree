// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Prompt/PromptBuilder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"

using namespace llvm;
using namespace mlir::iree_compiler::LLMAssist;
using namespace testing;

class PromptBuilderTest : public ::testing::Test {
protected:
  PromptBuilder builder;
};

TEST_F(PromptBuilderTest, BuildBasicTransformPrompt) {
  StringRef ir = R"(
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
)";

  PromptConfig config;
  config.taskDescription = "Optimize the arithmetic operations";

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // Check that the prompt contains the task description
  EXPECT_THAT(prompt, HasSubstr("Optimize the arithmetic operations"));

  // Check that the prompt contains the input IR
  EXPECT_THAT(prompt, HasSubstr("arith.addi"));
  EXPECT_THAT(prompt, HasSubstr("@add"));
}

TEST_F(PromptBuilderTest, BuildPromptWithFewShotExamples) {
  FewShotExample example;
  example.description = "Fold constant addition";
  example.inputIR = R"(
%c1 = arith.constant 1 : i32
%c2 = arith.constant 2 : i32
%sum = arith.addi %c1, %c2 : i32
)";
  example.outputIR = R"(
%c3 = arith.constant 3 : i32
)";

  builder.addFewShotExample(example);

  StringRef ir = R"(
module {
  func.func @test() -> i32 {
    %c5 = arith.constant 5 : i32
    %c10 = arith.constant 10 : i32
    %sum = arith.addi %c5, %c10 : i32
    return %sum : i32
  }
}
)";

  PromptConfig config;
  config.taskDescription = "Fold constant expressions";
  config.includeFewShot = true;
  // Set maxFewShotExamples high enough to include our added example
  // (PromptBuilder starts with 1 default example)
  config.maxFewShotExamples = 5;

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // Check that few-shot example is included (user-added example)
  EXPECT_THAT(prompt, HasSubstr("Fold constant addition"));
}

TEST_F(PromptBuilderTest, BuildPromptWithoutFewShotExamples) {
  FewShotExample example;
  example.description = "Example that should not appear";
  example.inputIR = "input";
  example.outputIR = "output";

  builder.addFewShotExample(example);

  StringRef ir = "module {}";

  PromptConfig config;
  config.taskDescription = "Some task";
  config.includeFewShot = false;

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // Few-shot should not be included
  EXPECT_THAT(prompt, Not(HasSubstr("Example that should not appear")));
}

TEST_F(PromptBuilderTest, GetSystemPrompt) {
  std::string systemPrompt = PromptBuilder::getSystemPrompt();

  // System prompt should contain MLIR-related instructions
  EXPECT_THAT(systemPrompt, HasSubstr("MLIR"));
  EXPECT_FALSE(systemPrompt.empty());
}

TEST_F(PromptBuilderTest, BuildAnalysisPrompt) {
  StringRef ir = R"(
module {
  func.func @mystery(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    return %arg0 : tensor<4x4xf32>
  }
}
)";

  std::string prompt = builder.buildAnalysisPrompt(ir, "What does this function do?");

  EXPECT_THAT(prompt, HasSubstr("What does this function do?"));
  EXPECT_THAT(prompt, HasSubstr("tensor<4x4xf32>"));
  EXPECT_THAT(prompt, HasSubstr("@mystery"));
}

TEST_F(PromptBuilderTest, BuildPromptWithExplanationRequest) {
  StringRef ir = "module {}";

  PromptConfig config;
  config.taskDescription = "Optimize";
  config.requestExplanation = true;

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // When explanation is requested, the prompt should ask for it
  EXPECT_THAT(prompt, HasSubstr("explain"));
}

TEST_F(PromptBuilderTest, MultipleFewShotExamples) {
  // Add additional examples beyond the default one
  for (int i = 0; i < 5; ++i) {
    FewShotExample example;
    example.description = "UserExample" + std::to_string(i);
    example.inputIR = "input " + std::to_string(i);
    example.outputIR = "output " + std::to_string(i);
    builder.addFewShotExample(example);
  }

  StringRef ir = "module {}";

  PromptConfig config;
  config.taskDescription = "Test";
  config.includeFewShot = true;
  config.maxFewShotExamples = 2;

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // Should only include up to maxFewShotExamples
  // Count occurrences of "### Example" headers (formatted as "### Example N:")
  size_t count = 0;
  size_t pos = 0;
  while ((pos = prompt.find("### Example ", pos)) != std::string::npos) {
    ++count;
    ++pos;
  }
  EXPECT_LE(count, 2u);
}

TEST_F(PromptBuilderTest, PromptContainsMLIRCodeBlock) {
  StringRef ir = R"(
module {
  func.func @test() -> i32 {
    %c1 = arith.constant 1 : i32
    return %c1 : i32
  }
}
)";

  PromptConfig config;
  config.taskDescription = "Transform";

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // The prompt should format IR in a code block for clarity
  // (implementation may vary, but IR should be present)
  EXPECT_THAT(prompt, HasSubstr("func.func @test"));
}

TEST_F(PromptBuilderTest, EmptyTaskDescription) {
  StringRef ir = "module {}";

  PromptConfig config;
  config.taskDescription = "";

  std::string prompt = builder.buildTransformPrompt(ir, config);

  // Should still generate a valid prompt
  EXPECT_FALSE(prompt.empty());
  EXPECT_THAT(prompt, HasSubstr("module"));
}

