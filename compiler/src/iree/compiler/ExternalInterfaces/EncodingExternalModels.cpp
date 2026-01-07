// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/EncodingExternalModels.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "iree-encoding-external-models"

namespace mlir::iree_compiler {
namespace {

/// Propagate an encoding through an "encoding castable" op. Encoding castable
/// means that the op can be encoded by casting its types to the encoded types.
/// This transform adds iree_encoding.set_encoding ops to the operands of the
/// `op`, and clones the `op` with the new encoded operands and encoded result
/// types. If the `propagationSource` comes froman iree_encoding.unset_encoding
/// op, and it is consumed by the `op`, then take the source of the unset
/// encoding instead of re-setting the encoding. If the `propagationSource` is
/// produced by the `op`, then do not unset the encoding after cloning the op,
/// because the encoded result will be used for propagation.
///
/// Use this function for ops that:
/// 1. Are encoded by casting their types to the encoded types.
/// 2. Are able to directly use the source of any producer unset_encoding ops
///    for propagation, and do not need to re-set the encoding.
static FailureOr<IREE::Encoding::PropagationResult>
propagateThroughEncodingCastableOp(
    RewriterBase &builder, Operation *op,
    IREE::Encoding::PropagationEncoding encodings,
    OpOperand *propagationSource) {
  SmallVector<Value> encodedOperands;
  IREE::Encoding::PropagationResult result;
  auto maybeUnsetEncodingProducer =
      propagationSource->get().getDefiningOp<IREE::Encoding::UnsetEncodingOp>();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  // Try to rematerialize encoding dims at the new insertion point.
  // For bubble-up propagation through encoding castable ops (tensor.cast,
  // tensor.collapse_shape, etc.), tensor.dim ops on the result can be
  // recreated from the source operand. These ops have a single tensor
  // input as their first operand.
  Value sourceOperand = op->getOperand(0);
  FailureOr<SmallVector<Value>> rematerializedDims =
      IREE::Encoding::rematerializeEncodingDims(
          builder, op, encodings.encodingDims, propagationSource->get(),
          sourceOperand);
  if (failed(rematerializedDims)) {
    return failure();
  }
  // Update encodings with rematerialized dims.
  encodings.encodingDims = *rematerializedDims;

  for (auto [operand, encoding] :
       llvm::zip_equal(op->getOperands(), encodings.operandEncodings)) {
    // Scalar operands do not need encodings.
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType) {
      encodedOperands.push_back(operand);
      continue;
    }
    // If the operand comes from the provided producer unset_encoding op, then
    // we don't need to set the encoding again, because this operand is the
    // source of propagation.
    if (operand == propagationSource->get() && maybeUnsetEncodingProducer) {
      encodedOperands.push_back(maybeUnsetEncodingProducer.getSource());
      continue;
    }
    auto encodedOperandType = operandType.cloneWithEncoding(encoding);
    // Special case for tensor.empty ops, which can simply be cloned with the
    // encoding, instead of creating a new set_encoding op.
    if (auto emptyOp = operand.getDefiningOp<tensor::EmptyOp>()) {
      auto encodedEmptyOp = tensor::EmptyOp::create(
          builder, op->getLoc(), encodedOperandType.getShape(),
          encodedOperandType.getElementType(), emptyOp.getDynamicSizes(),
          encoding);
      encodedOperands.push_back(encodedEmptyOp.getResult());
      continue;
    }
    // Otherwise, we need to create a new set_encoding op.
    auto setEncodingOp = IREE::Encoding::SetEncodingOp::create(
        builder, op->getLoc(), encodedOperandType, operand,
        encodings.encodingDims);
    encodedOperands.push_back(setEncodingOp.getResult());
    result.generatedEncodingOps.push_back(setEncodingOp);
  }
  SmallVector<Type> encodedResultTypes;
  for (auto [result, encoding] :
       llvm::zip_equal(op->getResults(), encodings.resultEncodings)) {
    auto resultType = cast<RankedTensorType>(result.getType());
    auto encodedResultType = resultType.cloneWithEncoding(encoding);
    encodedResultTypes.push_back(encodedResultType);
  }
  Operation *encodedOp =
      clone(builder, op, encodedResultTypes, encodedOperands);
  for (OpResult encodedResult : encodedOp->getOpResults()) {
    // If this encoded result is coming from the source of propagation, we want
    // to return the encoded result.
    OpResult originalResult = op->getOpResult(encodedResult.getResultNumber());
    if (originalResult == propagationSource->get()) {
      result.replacements.push_back(encodedResult);
      continue;
    }
    // Otherwise, we need to unset the encoding so the types are consistent with
    // the other results' users.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(builder, op->getLoc(), encodedResult);
    SmallVector<Value> resultDynamicDims;
    std::tie(std::ignore, resultDynamicDims) = decomposeMixedValues(mixedSizes);
    auto unsetEncodingOp = IREE::Encoding::UnsetEncodingOp::create(
        builder, op->getLoc(), originalResult.getType(), encodedResult,
        resultDynamicDims, encodings.encodingDims);
    result.generatedEncodingOps.push_back(unsetEncodingOp);
    result.replacements.push_back(unsetEncodingOp.getResult());
  }
  return result;
}

struct EncodingAttrPropagationInterface final
    : IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          EncodingAttrPropagationInterface, IREE::Encoding::EncodingAttr> {
  bool isPropagableDown(Attribute attr, OpOperand *target) const {
    return TypeSwitch<Operation *, bool>(target->getOwner())
        .Case<linalg::GenericOp>([&](auto genericOp) {
          // Only support parallel generic ops.
          if (genericOp.getNumReductionLoops() != 0) {
            return false;
          }
          // The unset encoding should not be on one of the inits.
          if (genericOp.isDpsInit(target)) {
            return false;
          }
          // Non-projected permutation indexing maps will unlikely get lowered
          // correctly with the encoding.
          if (llvm::any_of(genericOp->getOpOperands(), [&](OpOperand &operand) {
                AffineMap map = genericOp.getMatchingIndexingMap(&operand);
                return !map.isProjectedPermutation();
              })) {
            return false;
          }
          return true;
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateSinkingEncodings(Attribute attr, OpOperand *target) const {
    auto encoding = cast<IREE::Encoding::EncodingAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target->getOwner())
        .Case<linalg::GenericOp>([&](auto genericOp) {
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.operandEncodings.reserve(genericOp->getNumOperands());
          // Append the target and respective operand's indexing maps to the
          // encoding's indexing maps to create the new encoding.
          AffineMap invTargetIndexingMap = mlir::inversePermutation(
              genericOp.getMatchingIndexingMap(target));
          auto createNewEncoding =
              [&](AffineMap operandMap) -> IREE::Encoding::EncodingAttr {
            IREE::Encoding::EncodingAttr newEncoding = encoding;
            if (!invTargetIndexingMap.isIdentity()) {
              newEncoding = newEncoding.cloneWithNewOperandIndexingMap(
                  invTargetIndexingMap);
            }
            if (!operandMap.isIdentity()) {
              newEncoding =
                  newEncoding.cloneWithNewOperandIndexingMap(operandMap);
            }
            return newEncoding;
          };
          for (OpOperand *operand : genericOp.getDpsInputOperands()) {
            if (operand != target) {
              AffineMap operandMap = genericOp.getMatchingIndexingMap(operand);
              IREE::Encoding::EncodingAttr newEncoding =
                  createNewEncoding(operandMap);
              propEncoding.operandEncodings.push_back(newEncoding);
            } else {
              propEncoding.operandEncodings.push_back(encoding);
            }
          }
          for (OpOperand &operand : genericOp.getDpsInitsMutable()) {
            AffineMap operandMap = genericOp.getMatchingIndexingMap(&operand);
            IREE::Encoding::EncodingAttr newEncoding =
                createNewEncoding(operandMap);
            propEncoding.operandEncodings.push_back(newEncoding);
            propEncoding.resultEncodings.push_back(newEncoding);
          }
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct LayoutAttrPropagationInterface final
    : IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          LayoutAttrPropagationInterface, IREE::Encoding::LayoutAttr> {
  bool isPropagableUp(Attribute attr, OpResult target) const {
    auto layoutAttr = cast<IREE::Encoding::LayoutAttr>(attr);
    return TypeSwitch<Operation *, bool>(target.getOwner())
        .Case<tensor::CastOp>([&](auto castOp) {
          // CastOp is propagable if it is casting between compatible shapes,
          // because the dimensions need to be consistent with the
          // user_indexing_maps carried by the encoding. The tensor.cast op
          // verifier already guarantees that the shapes are compatible.
          return layoutAttr.isSerialized();
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateBubblingEncodings(Attribute attr, OpResult target) const {
    auto encoding = cast<IREE::Encoding::LayoutAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target.getOwner())
        .Case<tensor::CastOp>([&](tensor::CastOp) {
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.resultEncodings.push_back(encoding);
          propEncoding.operandEncodings.push_back(encoding);
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

template <typename OpTy>
struct EncodingCastableOpPropagationInterface final
    : IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          EncodingCastableOpPropagationInterface<OpTy>, OpTy> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &rewriter,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpOperand *propagationSource) const {
    return propagateThroughEncodingCastableOp(rewriter, op, encodings,
                                              propagationSource);
  }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated EncodingPropagationOpInterface.
template <typename... Ops>
struct EncodingCastableOpPropagationInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<EncodingCastableOpPropagationInterface<Ops>>(
         *context),
     ...);
  }
};

//===----------------------------------------------------------------------===//
// EncodingDimReificationInterface external models
//===----------------------------------------------------------------------===//

/// External model for tensor::CastOp to implement
/// EncodingDimReificationInterface. tensor.cast preserves encoding dims from
/// its source.
struct TensorCastEncodingDimModel
    : public IREE::Encoding::EncodingDimReificationInterface::ExternalModel<
          TensorCastEncodingDimModel, tensor::CastOp> {

  FailureOr<Value> reifyEncodingDim(Operation *op, OpBuilder &builder,
                                    unsigned resultIndex,
                                    unsigned dimIndex) const {
    // tensor.cast is a pass-through - we don't directly provide values.
    return failure();
  }

  Value getEncodingDimSource(Operation *op, unsigned resultIndex) const {
    auto castOp = cast<tensor::CastOp>(op);

    // Only result index 0 is valid for tensor.cast.
    if (resultIndex != 0)
      return {};

    // Check that the cast preserves the encoding.
    auto srcType = dyn_cast<RankedTensorType>(castOp.getSource().getType());
    auto dstType = dyn_cast<RankedTensorType>(castOp.getType());
    if (!srcType || !dstType)
      return {};

    if (srcType.getEncoding() != dstType.getEncoding())
      return {};

    return castOp.getSource();
  }
};

/// External model for tensor::EmptyOp to implement
/// EncodingDimReificationInterface. tensor.empty can directly provide encoding
/// dim values from its dynamic_sizes operands.
struct TensorEmptyEncodingDimModel
    : public IREE::Encoding::EncodingDimReificationInterface::ExternalModel<
          TensorEmptyEncodingDimModel, tensor::EmptyOp> {

  FailureOr<Value> reifyEncodingDim(Operation *op, OpBuilder &builder,
                                    unsigned resultIndex,
                                    unsigned dimIndex) const {
    auto emptyOp = cast<tensor::EmptyOp>(op);

    // Only result index 0 is valid for tensor.empty.
    if (resultIndex != 0)
      return failure();

    auto resultType = dyn_cast<RankedTensorType>(emptyOp.getType());
    if (!resultType)
      return failure();

    auto encoding =
        dyn_cast_or_null<IREE::Encoding::EncodingAttr>(resultType.getEncoding());
    if (!encoding)
      return failure();

    // For matmul encodings, encoding dimensions map directly to the iteration
    // dimensions (M, N, K). For the result tensor, we need to find which shape
    // dimension corresponds to the requested encoding dimension.
    //
    // Example: For a matmul result with iteration_sizes = [?, 128256, 4096]
    // and tensor<?x128256xf32>, encoding dim 0 (M dimension) corresponds to
    // shape dim 0.
    
    auto iterationSizes = encoding.getIterationSizes();
    if (!iterationSizes || dimIndex >= iterationSizes.size())
      return failure();

    // Map encoding dimension to shape dimension based on the operand index.
    // For matmul: LHS (operand 0) = [M, K], RHS (operand 1) = [N, K], Result
    // (operand 2) = [M, N]
    unsigned operandIndex = encoding.getOperandIndex().getValue().getZExtValue();
    
    ArrayRef<int64_t> shape = resultType.getShape();
    int64_t shapeDim = -1;
    
    if (operandIndex == 0) {
      // LHS: encoding dims [M, K] -> shape dims [0, 1]
      shapeDim = dimIndex;
    } else if (operandIndex == 1) {
      // RHS: encoding dims [N, K] -> shape dims [0, 1]
      shapeDim = dimIndex;
    } else if (operandIndex == 2) {
      // Result: encoding dims [M, N] -> shape dims [0, 1]
      // (K is reduction dimension, not in result shape)
      shapeDim = dimIndex;
    }
    
    if (shapeDim < 0 || shapeDim >= static_cast<int64_t>(shape.size()))
      return failure();

    // If the shape dimension is static, the encoding dimension is that static
    // value.
    if (!resultType.isDynamicDim(shapeDim)) {
      return arith::ConstantIndexOp::create(builder, op->getLoc(), shape[shapeDim]).getResult();
    }

    // Find the corresponding dynamic_sizes operand.
    unsigned dynamicDimIdx = 0;
    for (int64_t i = 0; i < shapeDim; ++i) {
      if (resultType.isDynamicDim(i))
        ++dynamicDimIdx;
    }

    if (dynamicDimIdx >= emptyOp.getDynamicSizes().size())
      return failure();

    return emptyOp.getDynamicSizes()[dynamicDimIdx];
  }

  Value getEncodingDimSource(Operation *op, unsigned resultIndex) const {
    // tensor.empty is a source operation - it doesn't forward encoding dims.
    return {};
  }
};

/// Fallback reification for any tensor value (including block arguments).
/// This creates a tensor.dim op as a last resort.
struct FallbackEncodingDimReification {
  static FailureOr<Value> reifyEncodingDim(OpBuilder &builder, Location loc,
                                             Value tensor, unsigned dimIndex) {
    auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
    if (!tensorType)
      return failure();

    auto encoding =
        dyn_cast_or_null<IREE::Encoding::EncodingAttr>(tensorType.getEncoding());
    if (!encoding)
      return failure();

    // Map encoding dimension to shape dimension
    auto iterationSizes = encoding.getIterationSizes();
    if (!iterationSizes || dimIndex >= iterationSizes.size())
      return failure();

    unsigned operandIndex = encoding.getOperandIndex().getValue().getZExtValue();
    ArrayRef<int64_t> shape = tensorType.getShape();
    int64_t shapeDim = -1;

    // Simple mapping for now: encoding dim -> shape dim (works for matmul)
    if (operandIndex == 0 || operandIndex == 1 || operandIndex == 2) {
      shapeDim = dimIndex;
    }

    if (shapeDim < 0 || shapeDim >= static_cast<int64_t>(shape.size()))
      return failure();

    // If static, return a constant
    if (!tensorType.isDynamicDim(shapeDim)) {
      return arith::ConstantIndexOp::create(builder, loc, shape[shapeDim])
          .getResult();
    }

    // For dynamic dimensions, create a tensor.dim op
    return tensor::DimOp::create(builder, loc, tensor, shapeDim).getResult();
  }
};

} // namespace

void registerEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::Encoding::IREEEncodingDialect *dialect) {
    IREE::Encoding::EncodingAttr::attachInterface<
        EncodingAttrPropagationInterface>(*ctx);
    IREE::Encoding::LayoutAttr::attachInterface<LayoutAttrPropagationInterface>(
        *ctx);
  });
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::tensor::TensorDialect *dialect) {
        EncodingCastableOpPropagationInterfaceHelper<
            tensor::CollapseShapeOp, tensor::CastOp>::registerOpInterface(ctx);
        // EncodingDimReificationInterface for tensor.cast and tensor.empty.
        tensor::CastOp::attachInterface<TensorCastEncodingDimModel>(*ctx);
        tensor::EmptyOp::attachInterface<TensorEmptyEncodingDimModel>(*ctx);
      });
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::linalg::LinalgDialect *dialect) {
        EncodingCastableOpPropagationInterfaceHelper<
            linalg::GenericOp>::registerOpInterface(ctx);
      });
}

} // namespace mlir::iree_compiler
