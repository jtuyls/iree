// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_IR_IRPARSER_H_
#define IREE_COMPILER_LLMASSIST_IR_IRPARSER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace mlir::iree_compiler::LLMAssist {

/// Result of parsing IR from text.
struct ParseResult {
  /// The parsed operation (if successful).
  OwningOpRef<ModuleOp> module;

  /// Diagnostic messages collected during parsing.
  std::vector<std::string> diagnostics;

  /// Whether parsing encountered errors.
  bool hadErrors = false;

  /// Check if parsing was successful.
  explicit operator bool() const { return !hadErrors && module; }
};

/// Utility class for parsing MLIR text back to IR.
class IRParser {
public:
  /// Parse a complete module from MLIR text.
  static ParseResult parseModule(llvm::StringRef irText, MLIRContext *ctx);

  /// Extract MLIR code blocks from LLM response text.
  /// Looks for ```mlir ... ``` blocks or raw MLIR content.
  static std::string extractMLIRFromResponse(llvm::StringRef response);

  /// Validate that the transformed module is structurally compatible with
  /// the original (same function signatures, etc.).
  static bool validateCompatibility(ModuleOp original, ModuleOp transformed);
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_IR_IRPARSER_H_

