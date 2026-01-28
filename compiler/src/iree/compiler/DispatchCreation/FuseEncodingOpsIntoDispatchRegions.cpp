// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEENCODINGOPSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Wraps an encoding op (SetEncodingOp or UnsetEncodingOp) in an
/// hoistable_dispatch op with implicit capture.
static FailureOr<IREE::Flow::HoistableDispatchOp>
wrapEncodingOpInDispatchRegion(RewriterBase &rewriter, Operation *encodingOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(encodingOp);
  Location loc = encodingOp->getLoc();

  // Get the input tensor and result type.
  Value source = encodingOp->getOperand(0);
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto resultType = cast<RankedTensorType>(encodingOp->getResult(0).getType());

  // Collect dynamic dims from the source type for input_dims.
  SmallVector<Value> inputDims;
  for (auto [idx, dim] : llvm::enumerate(sourceType.getShape())) {
    if (ShapedType::isDynamic(dim)) {
      inputDims.push_back(
          tensor::DimOp::create(rewriter, loc, source, idx).getResult());
    }
  }

  // Collect dynamic dims for the result type.
  SmallVector<Value> resultDims;
  for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
    if (ShapedType::isDynamic(dim)) {
      // For encoding ops, result dims come from the op itself or source.
      if (auto setEncoding =
              dyn_cast<IREE::Encoding::SetEncodingOp>(encodingOp)) {
        // SetEncodingOp preserves dims from source type.
        resultDims.push_back(
            tensor::DimOp::create(rewriter, loc, source, idx).getResult());
      } else if (auto unsetEncoding =
                     dyn_cast<IREE::Encoding::UnsetEncodingOp>(encodingOp)) {
        // UnsetEncodingOp has explicit result dims.
        unsigned dynamicIdx = resultType.getDynamicDimIndex(idx);
        if (dynamicIdx < unsetEncoding.getResultDims().size()) {
          resultDims.push_back(unsetEncoding.getResultDims()[dynamicIdx]);
        }
      }
    }
  }

  // Create the encoding dispatch region with the source as input.
  auto dispatchOp = IREE::Flow::HoistableDispatchOp::create(
      rewriter, loc, resultType, ValueRange{source}, inputDims, resultDims);

  // Create the body block (no block arguments - uses implicit capture).
  Block *body = rewriter.createBlock(&dispatchOp.getBody());
  rewriter.setInsertionPointToStart(body);

  // Clone the encoding op into the region (implicit capture of all operands).
  Operation *clonedOp = rewriter.clone(*encodingOp);

  // Return the result.
  IREE::Flow::ReturnOp::create(rewriter, loc, clonedOp->getResult(0));

  // Replace uses of the original op with the dispatch result.
  rewriter.replaceOp(encodingOp, dispatchOp.getResults());

  return dispatchOp;
}

/// Return true if the op is fusable with a SetEncodingOp consumer. The op's
/// containing dispatch must contain only:
///   - Reshape ops, encoding ops, linalg ops, gather ops, and attention ops.
///   - Non ShapedType ops, e.g., like arith ops, dim ops, etc.
///   - tensor::ExtractSliceOp is allowed as they can be folded into dispatch
///     tensor load ops.
static bool isFusableWithSetEncoding(Operation *target) {
  auto parentRegion = target->getParentOfType<IREE::Flow::DispatchRegionOp>();
  // Make sure the dispatch region has only one block.
  if (!llvm::hasSingleElement(parentRegion.getBody())) {
    return false;
  }
  // Check that there are no ops other than reshapes and element-wise linalg
  // ops in the dispatch region.
  Block &regionBlock = parentRegion.getBody().getBlocks().front();
  for (Operation &op : regionBlock.getOperations()) {
    if (llvm::none_of(op.getResultTypes(), llvm::IsaPred<ShapedType>)) {
      continue;
    }
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::EmptyOp,
             tensor::ExtractSliceOp, IREE::Encoding::SetEncodingOp,
             IREE::Encoding::UnsetEncodingOp, linalg::LinalgOp,
             IREE::LinalgExt::AttentionOp, IREE::LinalgExt::GatherOp>(op)) {
      return false;
    }
  }
  return true;
}

struct FuseEncodingOpsIntoDispatchRegionsPass final
    : impl::FuseEncodingOpsIntoDispatchRegionsPassBase<
          FuseEncodingOpsIntoDispatchRegionsPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    SmallVector<IREE::Encoding::SetEncodingOp> encodingOps;
    funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp)) {
        encodingOps.push_back(encodingOp);
      }
    });

    // First pass: try to fuse set_encoding ops with producer dispatches.
    for (IREE::Encoding::SetEncodingOp encodingOp : encodingOps) {
      OpOperand &operand = encodingOp.getSourceMutable();
      auto producerChain = getProducerDispatchValueAndOpChain(
          operand.get(), enableAggressiveFusion);
      if (!producerChain) {
        continue;
      }
      OpResult result = producerChain->first;
      auto producerDispatch =
          result.getDefiningOp<IREE::Flow::DispatchRegionOp>();
      auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
          producerDispatch.getBody().front().getTerminator());
      auto producerInRegion = dyn_cast<OpResult>(
          dispatchReturnOp->getOperand(result.getResultNumber()));
      if (!producerInRegion ||
          !isFusableWithSetEncoding(producerInRegion.getOwner())) {
        continue;
      }
      // Fuse the `encodingOp` and the producer chain into the dispatch.
      SmallVector<Operation *> dispatchConsumers(
          llvm::reverse(producerChain->second));
      dispatchConsumers.push_back(encodingOp);
      for (Operation *consumer : dispatchConsumers) {
        FailureOr<IREE::Flow::DispatchRegionOp> fusedDispatch =
            moveFollowingOpIntoDispatchRegion(rewriter, consumer,
                                              producerDispatch);
        if (failed(fusedDispatch)) {
          return signalPassFailure();
        }
        producerDispatch = fusedDispatch.value();
      }
    }

    // Second pass: wrap any remaining encoding ops in hoistable_dispatch.
    SmallVector<Operation *> remainingEncodingOps;
    funcOp->walk([&](Operation *op) {
      if (isa<IREE::Encoding::SetEncodingOp, IREE::Encoding::UnsetEncodingOp>(
              op) &&
          IREE::Flow::isNonNullAndOutsideDispatch(op)) {
        remainingEncodingOps.push_back(op);
      }
    });
    for (Operation *op : remainingEncodingOps) {
      if (failed(wrapEncodingOpInDispatchRegion(rewriter, op))) {
        return signalPassFailure();
      }
    }

    // Dynamic dims may have dominance issues after pulling encoding ops into
    // producer dispatch regions, so we need to resolve tensor.dim ops., Also
    // run the canonicalization patterns to remove redundantly returned results.
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    RewritePatternSet patterns(context);
    IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(patterns,
                                                              context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
