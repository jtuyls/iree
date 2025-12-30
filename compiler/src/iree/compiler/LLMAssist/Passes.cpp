// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/LLMAssist/Passes.h"

namespace mlir::iree_compiler::LLMAssist {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/LLMAssist/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerLLMAssistPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::LLMAssist

