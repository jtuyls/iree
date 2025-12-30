// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_IR_IRSERIALIZER_H_
#define IREE_COMPILER_LLMASSIST_IR_IRSERIALIZER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::LLMAssist {

/// Options for IR serialization.
struct SerializationOptions {
  /// Use generic MLIR form (more regular, better for LLM parsing).
  bool useGenericForm = false;

  /// Include location information.
  bool includeLocations = false;

  /// Use local scope for SSA names (simpler).
  bool useLocalScope = true;
};

/// Utility class for serializing MLIR operations to text.
class IRSerializer {
public:
  /// Serialize a full operation to string.
  static std::string serialize(Operation *op,
                               const SerializationOptions &opts = {});

  /// Serialize a module operation.
  static std::string serializeModule(ModuleOp module,
                                     const SerializationOptions &opts = {});
};

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_IR_IRSERIALIZER_H_

