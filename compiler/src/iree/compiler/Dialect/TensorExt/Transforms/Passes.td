// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOREXT_PASSES
#define IREE_DIALECT_TENSOREXT_PASSES

include "mlir/Pass/PassBase.td"

def FuseCollapseIntoStorePass :
    Pass<"iree-codegen-fuse-collapse-into-store", ""> {
  let summary = "TODOO";
}

#endif  // IREE_DIALECT_TENSOREXT_PASSES
