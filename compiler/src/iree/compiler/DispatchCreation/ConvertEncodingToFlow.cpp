// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "iree-dispatch-creation-convert-encoding-to-flow"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CONVERTENCODINGTOFLOWPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct ConvertEncodingToFlowPass
    : public impl::ConvertEncodingToFlowPassBase<ConvertEncodingToFlowPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void ConvertEncodingToFlowPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(&getContext());

  // Helper to check if op is inside a dispatch region.
  auto isInsideDispatchRegion = [](Operation *op) {
    return op->getParentOfType<IREE::Flow::DispatchRegionOp>() ||
           op->getParentOfType<IREE::Flow::HoistableDispatchOp>();
  };

  // Convert the set_encoding ops.
  funcOp.walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
    if (isInsideDispatchRegion(encodingOp)) {
      return;
    }
    rewriter.setInsertionPointAfter(encodingOp);
    Value source = encodingOp.getSource();
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, encodingOp.getLoc(), source);
    SmallVector<Value> dynamicDimSizes;
    std::tie(std::ignore, dynamicDimSizes) = decomposeMixedValues(mixedSizes);
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEncodeOp>(
        encodingOp, encodingOp.getResultType(), source,
        /*operand_dims=*/dynamicDimSizes, /*result_dims=*/dynamicDimSizes);
  });

  // Convert the unset_encoding ops.
  funcOp.walk([&](IREE::Encoding::UnsetEncodingOp encodingOp) {
    if (isInsideDispatchRegion(encodingOp)) {
      return;
    }
    rewriter.setInsertionPointAfter(encodingOp);
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEncodeOp>(
        encodingOp, encodingOp.getResultType(), encodingOp.getSource(),
        /*operand_dims=*/encodingOp.getResultDims(),
        /*result_dims=*/encodingOp.getResultDims());
  });
}

} // namespace mlir::iree_compiler::DispatchCreation
