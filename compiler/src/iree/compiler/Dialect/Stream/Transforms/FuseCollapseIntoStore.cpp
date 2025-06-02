// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-fuse-collapse-into-store"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_FUSECOLLAPSEINTOSTOREPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

// TODO: merge with same function on HAL interface binding ops.
struct FuseCollapseIntoTensorStoreOp
    : public OpRewritePattern<IREE::TensorExt::DispatchTensorStoreOp> {
  using OpRewritePattern<
      IREE::TensorExt::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::TensorExt::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "FuseCollapseIntoTensorStoreOp: " << storeOp << "\n");
    // if (!storeOp.isStoreToWholeTarget()) {
    //   return rewriter.notifyMatchFailure(storeOp, "unhandled partial
    //   stores");
    // }
    auto collapseOp = dyn_cast_if_present<tensor::CollapseShapeOp>(
        storeOp.getValue().getDefiningOp());
    if (!collapseOp) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected `tensor.collapse_shape` source");
    }
    RankedTensorType srcType = collapseOp.getSrcType();
    RankedTensorType collapseType = collapseOp.getResultType();
    SmallVector<ReassociationIndices, 4> reassociationMaps =
        collapseOp.getReassociationIndices();

    auto subspanOp = dyn_cast_if_present<IREE::Stream::BindingSubspanOp>(
        storeOp.getTarget().getDefiningOp());
    if (!subspanOp) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected `hal.interface.binding.subspan` target");
    }
    auto resultType = dyn_cast<IREE::TensorExt::DispatchTensorType>(
        subspanOp.getResult().getType());
    if (!resultType) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "expected result type to be !iree_tensor_ext.dispatch.tensor\n");
      return rewriter.notifyMatchFailure(
          subspanOp,
          "expected result type to be !iree_tensor_ext.dispatch.tensor");
    }

    Location loc = storeOp.getLoc();
    rewriter.setInsertionPoint(subspanOp);
    ArrayRef<int64_t> srcShape = srcType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    auto newResultType = IREE::TensorExt::DispatchTensorType::get(
        resultType.getAccess(), srcType);
    LLVM_DEBUG(llvm::dbgs() << "--newResultType: " << newResultType << "\n");
    SmallVector<Value> dynamicDims = subspanOp.getDynamicDims();
    LLVM_DEBUG(llvm::dbgs() << "--dynamicDims: " << dynamicDims.size() << "\n");
    SmallVector<Value> newSubspanDynamicDims;
    size_t dynIndex = 0;
    for (auto [i, dim] : llvm::enumerate(resultShape)) {
      if (!ShapedType::isDynamic(dim))
        continue;
      ReassociationIndices reassoc = reassociationMaps[i];
      int64_t staticVal = 1;
      bool foundDynamic = false;
      for (int64_t reassocIndex : reassoc) {
        int64_t srcDim = srcShape[reassocIndex];
        if (ShapedType::isDynamic(srcDim)) {
          if (foundDynamic)
            return failure();
          foundDynamic = true;
        } else {
          staticVal *= srcDim;
        }
      }
      AffineExpr result = rewriter.getAffineDimExpr(0).floorDiv(
          rewriter.getAffineConstantExpr(staticVal));
      AffineMap map = AffineMap::get(1, 0, result);

      auto newDynamicDim = rewriter.create<mlir::affine::AffineApplyOp>(
          loc, map, ValueRange{dynamicDims[dynIndex]});
      newSubspanDynamicDims.push_back(newDynamicDim);
      dynIndex++;
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::BindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getBinding(),
        subspanOp.getByteOffset(), newSubspanDynamicDims);
    // LLVM_DEBUG(llvm::dbgs() << "--success\n");

    LLVM_DEBUG(llvm::dbgs() << "BEFORE EXPAND\n");
    rewriter.setInsertionPoint(storeOp);

    // IREE::TensorExt::DispatchTensorType targetType = storeOp.getTargetType();
    // auto expandShapeType = IREE::TensorExt::DispatchTensorType::get(
    //   targetType.getAccess(), srcType);
    // auto expandShape = rewriter.create<tensor::ExpandShapeOp>(
    //   loc, expandShapeType, storeOp.getTarget(), reassociationMaps);

    // LLVM_DEBUG(llvm::dbgs() << "BEFORE OFFSETS\n");

    SmallVector<OpFoldResult> newOffsets(srcType.getRank(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> newStrides(srcType.getRank(),
                                         rewriter.getIndexAttr(1));
    // SmallVector<OpFoldResult> newMixedSizes =
    //     tensor::getMixedSizes(rewriter, loc, collapseOp.getSrc());
    SmallVector<OpFoldResult> newMixedSizes;
    size_t dynSizeIndex = 0;
    for (int64_t dim : srcShape) {
      if (ShapedType::isDynamic(dim)) {
        newMixedSizes.push_back(newSubspanDynamicDims[dynSizeIndex]);
        dynSizeIndex++;
      } else {
        newMixedSizes.push_back(rewriter.getIndexAttr(dim));
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "newMixedSizes: " << newMixedSizes.size() << "\n");
    SmallVector<int64_t> newStaticDims;
    SmallVector<Value> newDynamicDims;
    dispatchIndexOpFoldResults(newMixedSizes, newDynamicDims, newStaticDims);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorStoreOp>(
        storeOp, collapseOp.getSrc(), storeOp.getTarget(), newDynamicDims,
        newOffsets, newMixedSizes, newStrides);
    collapseOp->dropAllUses();
    rewriter.eraseOp(collapseOp);
    return success();
  }
};

struct FuseCollapseIntoStorePass final
    : impl::FuseCollapseIntoStorePassBase<FuseCollapseIntoStorePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto operation = getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "REWRITE STREAM FuseCollapseIntoTensorStoreOp\n");
    RewritePatternSet patterns(context);
    patterns.add<FuseCollapseIntoTensorStoreOp>(context);
    if (failed(applyPatternsGreedily(operation, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
