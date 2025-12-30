// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_PROMPT_PROMPTBUILDER_H_
#define IREE_COMPILER_LLMASSIST_PROMPT_PROMPTBUILDER_H_

#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// A few-shot example for prompting.
struct FewShotExample {
  std::string description;
  std::string inputIR;
  std::string outputIR;
};

/// Configuration for prompt building.
struct PromptConfig {
  /// The task description/instruction.
  std::string taskDescription;

  /// Whether to include few-shot examples.
  bool includeFewShot = true;

  /// Maximum number of few-shot examples to include.
  int maxFewShotExamples = 2;

  /// Whether to request the LLM to explain its changes.
  bool requestExplanation = false;
};

/// Builds prompts for LLM-assisted IR transformation.
class PromptBuilder {
public:
  PromptBuilder();

  /// Build a prompt for transforming the given IR.
  std::string buildTransformPrompt(llvm::StringRef irText,
                                   const PromptConfig &config) const;

  /// Build a prompt for analyzing/explaining IR (no transformation).
  std::string buildAnalysisPrompt(llvm::StringRef irText,
                                  llvm::StringRef question) const;

  /// Add a few-shot example.
  void addFewShotExample(const FewShotExample &example);

  /// Get the default system prompt.
  static std::string getSystemPrompt();

private:
  std::vector<FewShotExample> fewShotExamples_;

  /// Format few-shot examples into prompt text.
  std::string formatFewShotExamples(int maxExamples) const;
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_PROMPT_PROMPTBUILDER_H_

