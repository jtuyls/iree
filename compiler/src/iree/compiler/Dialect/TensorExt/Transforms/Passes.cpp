// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::TensorExt {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/TensorExt/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerTransformPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::IREE::TensorExt
