// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Prompt/PromptBuilder.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::LLMAssist {

PromptBuilder::PromptBuilder() {
  // Add some default few-shot examples for common transformations.
  addFewShotExample({
      "Optimize a simple addition by using a more efficient form",
      R"(module @example {
  func.func @add_one(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
})",
      R"(module @example {
  func.func @add_one(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
})"});
}

std::string PromptBuilder::getSystemPrompt() {
  return R"(You are an expert MLIR compiler engineer specializing in IREE (Intermediate Representation Execution Environment).

Your task is to transform MLIR code according to the given instructions while following these rules:

## Rules:
1. Output ONLY valid MLIR code inside a ```mlir code block
2. Preserve function signatures exactly (same name, same types)
3. Maintain SSA form - every value must be defined before use
4. Keep all type annotations correct and explicit
5. Do not introduce undefined values or operations
6. Preserve the semantics of the original code unless explicitly asked to change behavior

## IREE/MLIR Notes:
- Use `arith.` dialect for arithmetic operations
- Use `tensor.` dialect for tensor operations
- Use `linalg.` dialect for structured linear algebra
- Use `func.func` for function definitions
- Constants are defined with `arith.constant`
- Use `%name = operation` for SSA value definitions

## Output Format:
Wrap your transformed MLIR code in a ```mlir code block like this:
```mlir
module @name {
  // your code here
}
```)";
}

std::string PromptBuilder::buildTransformPrompt(llvm::StringRef irText,
                                                const PromptConfig &config) const {
  std::string prompt;
  llvm::raw_string_ostream os(prompt);

  // System prompt.
  os << getSystemPrompt() << "\n\n";

  // Few-shot examples if enabled.
  if (config.includeFewShot && !fewShotExamples_.empty()) {
    os << formatFewShotExamples(config.maxFewShotExamples) << "\n";
  }

  // Task description.
  os << "## Task:\n";
  os << config.taskDescription << "\n\n";

  // Input IR.
  os << "## Input MLIR:\n";
  os << "```mlir\n";
  os << irText << "\n";
  os << "```\n\n";

  // Request output.
  os << "## Output the transformed MLIR:\n";

  if (config.requestExplanation) {
    os << "(First briefly explain your changes, then provide the code)\n";
  }

  return prompt;
}

std::string PromptBuilder::buildAnalysisPrompt(llvm::StringRef irText,
                                               llvm::StringRef question) const {
  std::string prompt;
  llvm::raw_string_ostream os(prompt);

  os << "You are an expert MLIR compiler engineer. ";
  os << "Analyze the following MLIR code and answer the question.\n\n";

  os << "## MLIR Code:\n";
  os << "```mlir\n";
  os << irText << "\n";
  os << "```\n\n";

  os << "## Question:\n";
  os << question << "\n\n";

  os << "## Answer:\n";

  return prompt;
}

void PromptBuilder::addFewShotExample(const FewShotExample &example) {
  fewShotExamples_.push_back(example);
}

std::string PromptBuilder::formatFewShotExamples(int maxExamples) const {
  std::string result;
  llvm::raw_string_ostream os(result);

  os << "## Examples:\n\n";

  int count = 0;
  for (const auto &example : fewShotExamples_) {
    if (count >= maxExamples)
      break;

    os << "### Example " << (count + 1) << ": " << example.description << "\n\n";

    os << "Input:\n";
    os << "```mlir\n";
    os << example.inputIR << "\n";
    os << "```\n\n";

    os << "Output:\n";
    os << "```mlir\n";
    os << example.outputIR << "\n";
    os << "```\n\n";

    ++count;
  }

  return result;
}

} // namespace mlir::iree_compiler::LLMAssist

