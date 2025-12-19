// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingPatterns.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir::iree_compiler::IREE::Encoding {

//===----------------------------------------------------------------------===//
// Interface-based reification pattern
//===----------------------------------------------------------------------===//

/// Resolves iree_encoding.encoding_dim using EncodingDimReificationInterface.
///
/// This pattern handles operations that implement the interface in two ways:
/// 1. Operations that directly provide encoding dims (like set_encoding):
///    The pattern calls reifyEncodingDim() and replaces with the result.
/// 2. Operations that forward encoding dims from a source (like tensor.cast):
///    The pattern calls getEncodingDimSource() and creates a new encoding_dim
///    op on that source.
///
struct ReifyEncodingDimFromInterface : public OpRewritePattern<EncodingDimOp> {
  using OpRewritePattern::OpRewritePattern;

  // This pattern may create new EncodingDimOp, so we need to bound recursion.
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(EncodingDimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult result = dyn_cast<OpResult>(dimOp.getSource());
    if (!result) {
      return failure();
    }

    auto reificationOp =
        dyn_cast<EncodingDimReificationInterface>(result.getOwner());
    if (!reificationOp) {
      return failure();
    }

    int64_t dimIndex = dimOp.getConstantIndex();
    unsigned resultIndex = result.getResultNumber();

    // First, try direct reification (for ops like set_encoding).
    FailureOr<Value> directValue =
        reificationOp.reifyEncodingDim(rewriter, resultIndex, dimIndex);
    if (succeeded(directValue)) {
      rewriter.replaceOp(dimOp, *directValue);
      return success();
    }

    // Fall back to source tracing (for pass-through ops like tensor.cast).
    Value source = reificationOp.getEncodingDimSource(resultIndex);
    if (!source) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<EncodingDimOp>(dimOp, source,
                                               dimOp.getConstantIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Explicit pattern: Trace through DPS operations (linalg, etc.)
//===----------------------------------------------------------------------===//

/// For DPS operations where the result encoding comes from an init operand,
/// forward the dim query to that init operand.
///
/// This is an explicit pattern because DPS ops don't have a common base class
/// we can attach an external model to - they implement
/// DestinationStyleOpInterface which is an interface, not a concrete op.
///
/// Before:
///   %result = linalg.generic {outs(%init : tensor<?x?xf32, #enc>)} ...
///   %dim = iree_encoding.encoding_dim %result[0] : tensor<?x?xf32, #enc>
///
/// After:
///   %dim = iree_encoding.encoding_dim %init[0] : tensor<?x?xf32, #enc>
///
struct ReifyEncodingDimThroughDPS : public OpRewritePattern<EncodingDimOp> {
  using OpRewritePattern::OpRewritePattern;

  // This pattern creates a new EncodingDimOp, so we need to bound recursion.
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(EncodingDimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult result = dyn_cast<OpResult>(dimOp.getSource());
    if (!result) {
      return failure();
    }

    // Skip if already handled by interface.
    if (isa<EncodingDimReificationInterface>(result.getOwner())) {
      return failure();
    }

    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(result.getOwner());
    if (!dpsOp) {
      return failure();
    }

    // For DPS ops, the result encoding comes from the corresponding init.
    OpOperand *tiedInit = dpsOp.getTiedOpOperand(result);
    if (!tiedInit) {
      return failure();
    }

    // Verify encodings match.
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto initType = dyn_cast<RankedTensorType>(tiedInit->get().getType());
    if (!resultType || !initType) {
      return failure();
    }

    if (resultType.getEncoding() != initType.getEncoding()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<EncodingDimOp>(dimOp, tiedInit->get(),
                                               dimOp.getConstantIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fallback pattern: Reify encoding_dim using tensor.dim
//===----------------------------------------------------------------------===//

/// Fallback pattern that resolves iree_encoding.encoding_dim to tensor.dim
/// for sources that aren't OpResults (e.g., block arguments).
///
/// Before:
///   %dim = iree_encoding.encoding_dim %arg0[0] : tensor<?x?xf32, #enc>
///
/// After:
///   %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32, #enc>
///
struct ReifyEncodingDimToTensorDim : public OpRewritePattern<EncodingDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(EncodingDimOp dimOp,
                                PatternRewriter &rewriter) const override {
    // Only handle non-OpResult sources (e.g., block arguments).
    if (isa<OpResult>(dimOp.getSource())) {
      return failure();
    }

    auto tensorType = dyn_cast<RankedTensorType>(dimOp.getSource().getType());
    if (!tensorType)
      return failure();

    auto encoding =
        dyn_cast_or_null<IREE::Encoding::EncodingAttr>(tensorType.getEncoding());
    if (!encoding)
      return failure();

    // Map encoding dimension to shape dimension
    int64_t encodingDimIndex = dimOp.getConstantIndex();
    auto iterationSizes = encoding.getIterationSizes();
    if (!iterationSizes || encodingDimIndex >= static_cast<int64_t>(iterationSizes.size()))
      return failure();

    unsigned operandIndex = encoding.getOperandIndex().getValue().getZExtValue();
    int64_t shapeDim = -1;

    // Simple mapping: encoding dim -> shape dim (works for matmul)
    // For matmul: LHS (operand 0) = [M, K], RHS (operand 1) = [N, K], Result (operand 2) = [M, N]
    if (operandIndex == 0 || operandIndex == 1 || operandIndex == 2) {
      shapeDim = encodingDimIndex;
    }

    if (shapeDim < 0 || shapeDim >= tensorType.getRank())
      return failure();

    // If static, return a constant
    if (!tensorType.isDynamicDim(shapeDim)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          dimOp, tensorType.getDimSize(shapeDim));
      return success();
    }

    // For dynamic dimensions, create a tensor.dim op
    rewriter.replaceOpWithNewOp<tensor::DimOp>(dimOp, dimOp.getSource(),
                                                shapeDim);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateEncodingDimReificationPatterns(RewritePatternSet &patterns) {
  // Interface-based pattern (handles set_encoding, tensor.cast via external
  // model).
  patterns.add<ReifyEncodingDimFromInterface>(patterns.getContext());

  // Explicit pattern for DPS ops (interface-based approach not feasible for
  // DestinationStyleOpInterface since it's an interface, not a concrete op).
  patterns.add<ReifyEncodingDimThroughDPS>(patterns.getContext());

  // Fallback pattern for block arguments and other non-OpResult sources.
  patterns.add<ReifyEncodingDimToTensorDim>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::Encoding
