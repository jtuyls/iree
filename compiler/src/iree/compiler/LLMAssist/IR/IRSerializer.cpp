// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/IR/IRSerializer.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::LLMAssist {

std::string IRSerializer::serialize(Operation *op,
                                    const SerializationOptions &opts) {
  std::string result;
  llvm::raw_string_ostream os(result);

  OpPrintingFlags flags;
  if (opts.useLocalScope)
    flags.useLocalScope();
  if (!opts.includeLocations)
    flags.elideLargeElementsAttrs();
  if (opts.useGenericForm)
    flags.printGenericOpForm();

  op->print(os, flags);
  return result;
}

std::string IRSerializer::serializeModule(ModuleOp module,
                                          const SerializationOptions &opts) {
  return serialize(module.getOperation(), opts);
}

} // namespace mlir::iree_compiler::LLMAssist

