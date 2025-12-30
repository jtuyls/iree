// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_LLMASSIST_PASSES_H_
#define IREE_COMPILER_LLMASSIST_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::LLMAssist {

#define GEN_PASS_DECL
#include "iree/compiler/LLMAssist/Passes.h.inc"

/// Registers all LLMAssist passes.
void registerLLMAssistPasses();

} // namespace mlir::iree_compiler::LLMAssist

#endif // IREE_COMPILER_LLMASSIST_PASSES_H_

