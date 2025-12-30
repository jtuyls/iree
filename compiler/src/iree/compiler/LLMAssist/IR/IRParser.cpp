// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/IR/IRParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>

namespace mlir::iree_compiler::LLMAssist {

ParseResult IRParser::parseModule(llvm::StringRef irText, MLIRContext *ctx) {
  ParseResult result;

  // Set up diagnostic handler to collect messages.
  std::string diagnosticStr;
  llvm::raw_string_ostream diagnosticStream(diagnosticStr);

  ScopedDiagnosticHandler diagHandler(ctx, [&](Diagnostic &diag) {
    diagnosticStream << diag.str() << "\n";
    result.diagnostics.push_back(diag.str());
    if (diag.getSeverity() == DiagnosticSeverity::Error) {
      result.hadErrors = true;
    }
    return success();
  });

  // Parse the module.
  result.module = parseSourceString<ModuleOp>(irText, ctx);

  if (!result.module) {
    result.hadErrors = true;
    if (result.diagnostics.empty()) {
      result.diagnostics.push_back("Failed to parse MLIR");
    }
  }

  return result;
}

std::string IRParser::extractMLIRFromResponse(llvm::StringRef response) {
  std::string responseStr = response.str();

  // Try to find ```mlir ... ``` code blocks.
  std::regex mlirBlockRegex(R"(```(?:mlir)?\s*\n([\s\S]*?)```)");
  std::smatch match;

  if (std::regex_search(responseStr, match, mlirBlockRegex) &&
      match.size() > 1) {
    std::string extracted = match[1].str();
    // Trim leading/trailing whitespace.
    size_t start = extracted.find_first_not_of(" \t\n\r");
    size_t end = extracted.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
      return extracted.substr(start, end - start + 1);
    }
    return extracted;
  }

  // Fallback: look for module { ... } or func.func patterns.
  std::regex moduleRegex(R"((module\s*(?:@\w+)?\s*\{[\s\S]*\}))");
  if (std::regex_search(responseStr, match, moduleRegex) && match.size() > 1) {
    return match[1].str();
  }

  std::regex funcRegex(R"((func\.func\s+@\w+[\s\S]*))");
  if (std::regex_search(responseStr, match, funcRegex) && match.size() > 1) {
    // Wrap in a module.
    return "module {\n" + match[1].str() + "\n}";
  }

  // If nothing matched, return the original (let parsing fail with a good
  // error).
  return responseStr;
}

bool IRParser::validateCompatibility(ModuleOp original, ModuleOp transformed) {
  // For now, just check that we have the same number of functions.
  // More sophisticated validation could check:
  // - Function signatures match
  // - Types are compatible
  // - No undefined values

  // Collect functions from both modules.
  llvm::SmallVector<mlir::func::FuncOp> origVec;
  llvm::SmallVector<mlir::func::FuncOp> transVec;

  for (auto func : original.getOps<mlir::func::FuncOp>()) {
    origVec.push_back(func);
  }
  for (auto func : transformed.getOps<mlir::func::FuncOp>()) {
    transVec.push_back(func);
  }

  if (origVec.size() != transVec.size()) {
    return false;
  }

  // Check that function names and signatures match.
  for (size_t i = 0; i < origVec.size(); ++i) {
    if (origVec[i].getName() != transVec[i].getName()) {
      return false;
    }
    if (origVec[i].getFunctionType() != transVec[i].getFunctionType()) {
      return false;
    }
  }

  return true;
}

} // namespace mlir::iree_compiler::LLMAssist

