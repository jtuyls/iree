// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingPatterns.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Encoding {

//===----------------------------------------------------------------------===//
// encoding.set_encoding
//===----------------------------------------------------------------------===//

LogicalResult SetEncodingOp::verify() {
  // Source and the result have the same rank.
  if (getSourceType().getEncoding()) {
    return emitOpError(
        "source of set_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<SerializableAttr>(getResultType().getEncoding())) {
    return emitOpError(
        "result of set_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  return success();
}

LogicalResult SetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      tensor::getMixedSizes(builder, getLoc(), getSource());
  return success();
}

FailureOr<Value> SetEncodingOp::reifyEncodingDim(OpBuilder &builder,
                                                 unsigned resultIndex,
                                                 unsigned dimIndex) {
  // SetEncodingOp has a single result, so resultIndex must be 0.
  if (resultIndex != 0)
    return failure();

  ValueRange encodingDims = getEncodingDims();
  if (dimIndex >= encodingDims.size())
    return failure();

  return encodingDims[dimIndex];
}

//===----------------------------------------------------------------------===//
// encoding.unset_encoding
//===----------------------------------------------------------------------===//

LogicalResult UnsetEncodingOp::verify() {
  if (getResultType().getEncoding()) {
    return emitOpError(
        "result of unset_encoding op cannot have a tensor encoding");
  }
  if (!isa_and_nonnull<SerializableAttr>(getSourceType().getEncoding())) {
    return emitOpError(
        "source of unset_encoding op expected to have a valid tensor encoding");
  }
  // The source and result must have the same rank.
  if (getResultType().getRank() != getSourceType().getRank()) {
    return emitOpError("cannot change the rank of the tensor");
  }
  if (failed(verifyCompatibleShape(getResultType(), getSourceType()))) {
    return emitOpError("expected to preserve the logical shape of the tensor");
  }
  unsigned requiredDynCount = getResultType().getNumDynamicDims();
  if (getResultDims().size() != requiredDynCount) {
    return emitOpError() << "result type set has " << requiredDynCount
                         << " dynamic dimensions but only "
                         << getResultDims().size()
                         << " dimension values are attached";
  }
  return success();
}

LogicalResult UnsetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] =
      getMixedValues(getResultType().getShape(), getResultDims(), builder);
  return success();
}

//===----------------------------------------------------------------------===//
// encoding.encoding_dim
//===----------------------------------------------------------------------===//

LogicalResult EncodingDimOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  Attribute encoding = sourceType.getEncoding();

  if (!encoding) {
    return emitOpError("source tensor must have an encoding");
  }

  auto serializableAttr = dyn_cast<SerializableAttr>(encoding);
  if (!serializableAttr) {
    return emitOpError(
        "source tensor encoding must implement SerializableAttr");
  }

  // Check that the index is valid if we can determine the number of dims.
  std::optional<unsigned> numEncodingDims =
      serializableAttr.getNumEncodingDims();
  if (numEncodingDims) {
    int64_t index = getConstantIndex();
    if (index < 0 || static_cast<unsigned>(index) >= *numEncodingDims) {
      return emitOpError("encoding dimension index ")
             << index << " is out of bounds for encoding with "
             << *numEncodingDims << " dimensions";
    }
  }

  return success();
}

void EncodingDimOp::build(OpBuilder &builder, OperationState &result,
                          Value source, int64_t index) {
  build(builder, result, builder.getIndexType(), source,
        builder.getIndexAttr(index));
}

void EncodingDimOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "enc_dim");
}

/// Fold encoding_dim when the source comes from set_encoding with known
/// encoding_dims.
OpFoldResult EncodingDimOp::fold(FoldAdaptor adaptor) {
  int64_t index = getConstantIndex();

  // Trace through the producer chain to find set_encoding.
  Value source = getSource();
  while (source) {
    if (auto setEncoding = source.getDefiningOp<SetEncodingOp>()) {
      ValueRange encodingDims = setEncoding.getEncodingDims();
      if (index >= 0 && static_cast<size_t>(index) < encodingDims.size()) {
        // Check if the encoding_dim is a constant.
        if (auto constOp =
                encodingDims[index].getDefiningOp<arith::ConstantIndexOp>()) {
          return constOp.getValue();
        }
        // Not a constant, but we found the value - can't fold further.
        return {};
      }
      return {};
    }

    // Forward through tensor.cast.
    if (auto castOp = source.getDefiningOp<tensor::CastOp>()) {
      source = castOp.getSource();
      continue;
    }

    // No further producers to trace.
    break;
  }

  return {};
}

void EncodingDimOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  populateEncodingDimReificationPatterns(results);
}

} // namespace mlir::iree_compiler::IREE::Encoding

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.cpp.inc" // IWYU pragma: keep
