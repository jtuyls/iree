// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

static Type getComplexElementTypeOrSelf(Type ty) {
  if (auto complex = dyn_cast_or_null<ComplexType>(ty)) {
    return complex.getElementType();
  }
  return ty;
}

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange inputOperands, ValueRange outputOperands) {
  for (Value value : inputOperands) {
    if (!llvm::isa<MemRefType>(value.getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputOperands) {
    if (!llvm::isa<MemRefType>(value.getType())) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}

/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate. b) At least one dimension is out of bound (`dimPos` is >= 0 and <
/// rank). c) the number of elements in `dimsPos` is > than `rank`.
static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  // early exit.
  if (dimsPos.size() > rank) {
    return true;
  }
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos) {
    uniqued.insert(dim);
  }
  if (dimsPos.size() != uniqued.size()) {
    return true;
  }
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

/// Returns true if the dimension of `sourceShape` is smaller than the dimension
/// of the `limitShape`.
static bool isSmallerThan(ArrayRef<int64_t> sourceShape,
                          ArrayRef<int64_t> limitShape) {
  assert(
      sourceShape.size() == limitShape.size() &&
      "expected source shape rank, and limit of the shape to have same rank");
  return llvm::all_of(llvm::zip_equal(sourceShape, limitShape),
                      [](std::tuple<int64_t, int64_t> it) {
                        int64_t sourceExtent = std::get<0>(it);
                        int64_t limit = std::get<1>(it);
                        return ShapedType::isDynamic(sourceExtent) ||
                               ShapedType::isDynamic(limit) ||
                               sourceExtent <= limit;
                      });
}

/// Returns the size and offset scaled by some scale factor, and clamped to a
/// dimSize for the dimension. `(offset + size) * scale` will be clamped to the
/// `dimSize`.
static std::pair<OpFoldResult, OpFoldResult>
getScaledSizeAndOffset(OpBuilder &builder, Location loc, OpFoldResult size,
                       OpFoldResult offset, OpFoldResult dimSize,
                       int64_t offsetScale, int64_t sizeScale) {
  AffineExpr dim0, dim1, dim2;
  auto ctx = builder.getContext();
  bindDims(ctx, dim0, dim1, dim2);
  auto imageOffset = affine::makeComposedFoldedAffineApply(
      builder, loc, {dim0 * offsetScale}, offset);
  auto dimSizeValue = getValueOrCreateConstantIndexOp(builder, loc, dimSize);
  AffineMap sizeMap =
      AffineMap::get(3, 0, {dim0 - dim1, dim2 * sizeScale}, ctx);
  auto imageSize = affine::makeComposedFoldedAffineMin(
      builder, loc, sizeMap, {dimSizeValue, imageOffset, size});
  return std::make_pair(imageSize, imageOffset);
}

/// If the input has a fully static shape, return the static sizes. Otherwise,
/// attempt to reify the shape of the input from its defining op. Input dims
/// are store into `reifiedInputDims`.
static LogicalResult
getStaticOrReifiedInputDims(OpBuilder &builder, Location loc, Value input,
                            ReifiedRankedShapedTypeDims &reifiedInputDims) {
  if (auto reifyOp = input.getDefiningOp<ReifyRankedShapedTypeOpInterface>()) {
    return reifyOp.reifyResultShapes(builder, reifiedInputDims);
  }
  auto inputType = cast<ShapedType>(input.getType());
  if (!inputType.hasStaticShape()) {
    return failure();
  }
  reifiedInputDims.push_back(
      getAsIndexOpFoldResult(builder.getContext(), inputType.getShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verify() {
  Operation *op = getOperation();
  if (getInputs().size() != 2) {
    return op->emitOpError("expected two input operands");
  }
  if (getOutputs().size() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto checkDimensionsMatch = [&](ShapedType t1, ShapedType t2, unsigned dim) {
    return t1.getShape()[dim] == t2.getShape()[dim];
  };

  auto indicesType = getIndicesType();
  if (indicesType.getRank() != 2 ||
      !indicesType.getElementType().isInteger(32)) {
    return op->emitOpError(
        "expected indices to be of rank 2 of i32 element type");
  }
  auto indexDepth = getIndexDepth();
  if (ShapedType::isDynamic(indexDepth)) {
    return op->emitOpError("expected index depth is static");
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();
  if (dimMap.size() != indexDepth) {
    return op->emitOpError("invalid number of dimension map entries ");
  }

  auto originalType = getOriginalType();
  if (isInvalid(dimMap, originalType.getRank())) {
    return op->emitOpError("dimension map is invalid");
  }

  // The first dimension of the indices should match the first dimension of the
  // output. They indicate to the number of updates.
  auto updateType = getUpdateType();
  if (updateType.getRank() < 1) {
    return op->emitOpError("expected update value to be at least rank 1");
  }
  if (!checkDimensionsMatch(indicesType, updateType, 0)) {
    return op->emitOpError(
        "mismatch in shape of indices and update value at dim#0");
  }
  if (updateType.getRank() - 1 > originalType.getRank()) {
    return op->emitOpError(
        "update value rank exceeds the rank of the original value");
  }

  // indexDepth + update dims should cover the original dims. The first dim of
  // update is the number of updates.
  if (originalType.getRank() > indexDepth + updateType.getRank() - 1) {
    return op->emitOpError(
        "index depth and update value does not cover rank of original value");
  }

  // Validate the non-indexed update dims cover the full slice size of the
  // original tensor.
  int64_t fullSliceDims = originalType.getRank() - indexDepth;
  for (auto it :
       llvm::zip_equal(llvm::seq<unsigned>(indexDepth, originalType.getRank()),
                       llvm::seq<unsigned>(updateType.getRank() - fullSliceDims,
                                           updateType.getRank()))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim;
    }
  }

  // Check that the remaining update indices do not exceed the update length.
  int64_t insertDims = originalType.getRank() - updateType.getRank() + 1;
  for (auto it : llvm::zip_equal(
           llvm::seq<unsigned>(insertDims, indexDepth),
           llvm::seq<unsigned>(1, updateType.getRank() - fullSliceDims))) {
    int64_t originalDim = std::get<0>(it);
    int64_t updateDim = std::get<1>(it);
    if (!originalType.isDynamicDim(originalDim) &&
        updateType.getDimSize(updateDim) >
            originalType.getDimSize(originalDim)) {
      return op->emitOpError("indexed shape of update value dim#")
             << updateDim << " exceeds original value at dim#" << originalDim
             << " " << updateType.getDimSize(updateDim) << " "
             << originalType.getDimSize(originalDim);
    }
  }

  Region &region = this->getRegion();
  Block *body = &region.front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected region to have two arguments");
  }
  Type arg0Type = body->getArgument(0).getType();
  Type arg1Type = body->getArgument(1).getType();
  if (!getComplexElementTypeOrSelf(arg0Type).isIntOrFloat() ||
      !getComplexElementTypeOrSelf(arg1Type).isIntOrFloat()) {
    return op->emitOpError(
        "expected region to have scalar argument of integer or float types");
  }
  if (arg0Type != updateType.getElementType()) {
    return op->emitOpError("mismatch in argument 0 of region ")
           << arg0Type << " and element type of update value "
           << updateType.getElementType();
  }
  if (arg1Type != originalType.getElementType()) {
    return op->emitOpError("mismatch in argument 1 of region ")
           << arg1Type << " and element type of original value "
           << originalType.getElementType();
  }
  if (arg0Type != arg1Type) {
    return op->emitOpError("mismatch in region argument types ")
           << arg0Type << " and " << arg1Type;
  }
  auto yieldOp = cast<IREE::LinalgExt::YieldOp>(body->getTerminator());
  if (yieldOp->getNumOperands() != 1) {
    return yieldOp.emitOpError("expected region to yield a single value");
  }
  auto yieldedType = yieldOp->getOperand(0).getType();
  if (yieldedType != arg0Type) {
    return yieldOp.emitOpError("mismatch in type of yielded value ")
           << yieldedType << " and argument of the region " << arg0Type;
  }
  return success();
}

SmallVector<utils::IteratorType> ScatterOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getUpdateType().getRank(),
                                                 utils::IteratorType::parallel);
  if (!getUniqueIndices()) {
    iteratorTypes[0] = utils::IteratorType::reduction;
  }
  return iteratorTypes;
}

SmallVector<Range> ScatterOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getUpdateType().getRank())) {
    Value ub = getDimValue(builder, loc, updates(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

FailureOr<TilingResult>
ScatterOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  assert(offsets.size() >= 1 && sizes.size() >= 1);
  Location loc = getLoc();
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  // Slice of the updates.
  auto updateRank = getUpdateType().getRank();
  SmallVector<OpFoldResult> updateStrides(updateRank, oneAttr);
  Value tiledUpdate =
      getSlice(builder, loc, updates(), offsets, sizes, updateStrides);
  assert(tiledUpdate && "failed to get slice of update");

  // Slice of indices.
  auto indicesRank = getIndicesType().getRank();
  SmallVector<OpFoldResult> indicesOffsets(indicesRank, zeroAttr);
  SmallVector<OpFoldResult> indicesSizes(indicesRank);
  indicesOffsets[0] = offsets[0];
  indicesSizes[0] = sizes[0];
  for (auto dim : llvm::seq<int64_t>(1, indicesRank)) {
    indicesSizes[dim] = getDim(builder, loc, indices(), dim);
  }
  SmallVector<OpFoldResult> indicesStrides(indicesRank, oneAttr);
  Value tiledIndices = getSlice(builder, loc, indices(), indicesOffsets,
                                indicesSizes, indicesStrides);
  assert(tiledIndices && "failed to get slice of indices");

  // Slice of the original.
  SmallVector<OpFoldResult> originalOffsets, originalSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, originalOffsets,
                                   originalSizes))) {
    return {};
  }
  auto originalRank = getOriginalType().getRank();
  SmallVector<OpFoldResult> originalStrides(originalRank, oneAttr);
  Value tiledOriginal = getSlice(builder, loc, original(), originalOffsets,
                                 originalSizes, originalStrides);
  assert(tiledOriginal && "failed to get slice of original tensor");

  SmallVector<Type> resultTypes;
  if (getNumResults()) {
    resultTypes.push_back(tiledOriginal.getType());
  }
  Operation *tiledScatterOp =
      mlir::clone(builder, getOperation(), resultTypes,
                  ValueRange{tiledUpdate, tiledIndices, tiledOriginal});
  return TilingResult{{tiledScatterOp},
                      SmallVector<Value>(tiledScatterOp->getResults())};
}

LogicalResult ScatterOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  auto zeroAttr = builder.getI64IntegerAttr(0);
  // Slice of the original.
  auto originalRank = getOriginalType().getRank();
  resultOffsets.resize(originalRank, zeroAttr);
  resultSizes.resize(originalRank);

  auto updateRank = getUpdateType().getRank();
  Location loc = getLoc();
  for (auto dim : llvm::seq<int64_t>(0, originalRank - updateRank + 1)) {
    resultSizes[dim] = getDim(builder, loc, original(), dim);
  }
  for (auto dim :
       llvm::seq<int64_t>(originalRank - updateRank + 1, originalRank)) {
    resultOffsets[dim] = offsets[dim - (originalRank - updateRank)];
    resultSizes[dim] = sizes[dim - (originalRank - updateRank)];
  }
  return success();
}

LogicalResult ScatterOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  auto indexDepth = getIndexDepth();
  Value update = b.create<memref::LoadOp>(loc, updates(), ivs);
  SmallVector<Value> starts;
  SmallVector<Value> loadIndices;
  loadIndices.push_back(ivs.front());
  loadIndices.push_back(Value());

  // Populate with empty values.
  auto originalTy = cast<ShapedType>(original().getType());
  starts.resize(originalTy.getRank(), Value());
  auto updateIvs = ivs.drop_front(1);

  int64_t offset = starts.size() - updateIvs.size();
  for (auto [idx, iv] : llvm::enumerate(updateIvs)) {
    starts[idx + offset] = iv;
  }

  ArrayRef<int64_t> dimMap = getDimensionMap();

  for (auto i : llvm::seq<unsigned>(0, indexDepth)) {
    loadIndices.back() = b.create<arith::ConstantIndexOp>(loc, i);
    Value idx = b.create<memref::LoadOp>(loc, indices(), loadIndices);
    Value ret = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);

    auto dim = dimMap[i];

    if (starts[dim])
      ret = b.create<arith::AddIOp>(loc, ret, starts[dim]);
    starts[dim] = ret;
  }

  Value init = b.create<memref::LoadOp>(loc, original(), starts);

  IRMapping bvm;
  Block &block = getRegion().front();
  bvm.map(block.getArgument(0), update);
  bvm.map(block.getArgument(1), init);
  for (auto &blockOp : block.without_terminator()) {
    b.clone(blockOp, bvm);
  }
  // The last op is linalg_ext.yield op. Store the operand to
  // destination.
  b.create<memref::StoreOp>(
      loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)),
      original(), starts);
  return success();
}

LogicalResult
ScatterOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

LogicalResult SortOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs()) {
    return op->emitOpError("does not expect to take any inputs");
  }
  if (getNumDpsInits() == 0) {
    return op->emitOpError("expected at least one `outs` operand");
  }

  Block &block = getRegion().front();
  size_t numOutputs = getNumDpsInits();
  if (block.getNumArguments() != 2 * numOutputs) {
    return op->emitOpError("region block should have ")
           << 2 * numOutputs << " arguments";
  }

  int64_t rank = getOperandRank();
  int sortDim = getDimension();
  if (sortDim < 0 || sortDim >= rank) {
    return op->emitOpError("dimension must be within (0, ") << rank << "]";
  }

  ArrayRef<int64_t> shape = getOperandShape();
  for (auto [index, operand] : llvm::enumerate(getOutputs())) {
    auto operandType = getOperandType(index);
    if (operandType.getRank() != rank) {
      return op->emitOpError("expected operand ")
             << index << " to be rank " << rank << ", same as other operands";
    }
    if (operandType.getShape() != shape) {
      return op->emitOpError("expected operand ")
             << index << " to have same shape as other operands";
    }
    Type elemType = operandType.getElementType();
    for (int i : {2 * index, 2 * index + 1}) {
      Type argType = block.getArgument(i).getType();
      if (argType != elemType) {
        return op->emitOpError("region block argument #")
               << i << " should be of type " << elemType << " but got "
               << argType;
      }
    }
  }

  auto yieldOp = cast<YieldOp>(block.getTerminator());
  if (yieldOp.getNumOperands() != 1) {
    return op->emitOpError("should yield exactly one operand");
  }
  auto ty = yieldOp.getOperand(0).getType().dyn_cast<IntegerType>();
  if (!ty || ty.getWidth() != 1) {
    return op->emitOpError("should yield i1 type");
  }

  return success();
}

SmallVector<utils::IteratorType> SortOp::getLoopIteratorTypes() {
  // All loops except the dimension to sort along are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

SmallVector<Range> SortOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = operand(0);
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

FailureOr<TilingResult>
SortOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands(getOutputs().size());
  for (auto [idx, output] : llvm::enumerate(getOutputs())) {
    tiledOperands[idx] =
        getSlice(builder, getLoc(), output, offsets, sizes, strides);
    assert(tiledOperands[idx] && "failed to get slice of operand");
  }
  SmallVector<Type, 4> resultTypes;
  if (getNumResults()) {
    resultTypes = llvm::map_to_vector<4>(tiledOperands,
                                         [&](Value v) { return v.getType(); });
  }
  Operation *tiledSortOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledSortOp},
                      SmallVector<Value>{tiledSortOp->getResults()}};
}

LogicalResult SortOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

LogicalResult SortOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  auto sortDim = getDimension();
  SmallVector<Value> indices, sortBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  // Bubble sort innermost loop.
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value ub;
  if (getOperandType(0).isDynamicDim(sortDim)) {
    ub = b.create<memref::DimOp>(loc, operand(0), sortDim);
  } else {
    ub = b.create<arith::ConstantIndexOp>(
        loc, getOperandType(0).getDimSize(sortDim));
  }
  ub = b.create<arith::SubIOp>(loc, ub, one);
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iters) {
        SmallVector<Value> indices(ivs);
        Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
        for (auto output : getDpsInits()) {
          indices[sortDim] = iv;
          sortBlkArgs.push_back(b.create<memref::LoadOp>(loc, output, indices));
          indices[sortDim] = ivPlusOne;
          sortBlkArgs.push_back(b.create<memref::LoadOp>(loc, output, indices));
        }
      });

  auto &srcBlock = getRegion().front();
  Region &region = scfFor.getRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), sortBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
  }
  Value cond = bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0));

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToEnd(&region.front());
  b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        // Do not swap the pairs if true.
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        // Swap the pairs if false.
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value ivPlusOne =
            b.create<arith::AddIOp>(loc, scfFor.getInductionVar(), one);
        for (int i = 0, e = getNumDpsInits(); i < e; ++i) {
          Value v1 = sortBlkArgs[i * 2];
          Value v2 = sortBlkArgs[i * 2 + 1];
          indices[sortDim] = scfFor.getInductionVar();
          b.create<memref::StoreOp>(loc, v2, getDpsInits()[i], indices);
          indices[sortDim] = ivPlusOne;
          b.create<memref::StoreOp>(loc, v1, getDpsInits()[i], indices);
        }
        b.create<scf::YieldOp>(loc);
      });
  b.create<scf::YieldOp>(loc);
  return success();
}

LogicalResult
SortOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

LogicalResult FftOp::verify() {
  Operation *op = getOperation();
  auto length = getFftLength();
  // After tiling, it could be dynamic shape. (Because
  // subview/subtensor does not inference the type correctly
  // on (1 << x)) cases).
  if (ShapedType::isDynamic(length))
    return success();
  if (length & (length - 1)) {
    return op->emitOpError("only powers of 2 are handled currently");
  }
  if (!getNumDpsInputs() || !isScalar(getDpsInputOperand(0))) {
    return op->emitOpError("expected to carry `stage` input");
  }
  if (getNumDpsInputs() != 1) {
    if (getNumDpsInputs() != 3 || isScalar(getDpsInputOperand(1)) ||
        isScalar(getDpsInputOperand(2))) {
      return op->emitOpError("expected to carry real and imag coeff inputs");
    }
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError(
        "expected outputs to be real and imag tensor/memref");
  }
  return success();
}

SmallVector<utils::IteratorType> FftOp::getLoopIteratorTypes() {
  // There are `rank-1` outer loops. The fft itselfs has one loop for each
  // stage, which handles the merge step -- taking two half size tensors and
  // merge them into one tensor.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

SmallVector<Range> FftOp::getIterationDomain(OpBuilder &builder) {
  SmallVector<Range> res;
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (auto [idx, val] : llvm::enumerate(getOperandShape().drop_back())) {
    Value size;
    if (ShapedType::isDynamic(val)) {
      size = getDimValue(builder, loc, getReal(), idx);
    } else {
      size = builder.create<arith::ConstantIndexOp>(loc, val);
    }
    res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/one});
  }

  Value size = getDimValue(builder, loc, getReal(), getOperandRank() - 1);
  Value stride = builder.create<arith::ShLIOp>(loc, one, getStage());
  res.emplace_back(Range{/*offset=*/zero, size, /*stride=*/stride});
  return res;
}

void FftOp::generateScalarImplWithoutCoeffBuf(OpBuilder &b, Location loc,
                                              ArrayRef<Value> operands,
                                              Value wholeSize) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps(operands.size(), b.getMultiDimIdentityMap(rank));

  auto f32Type = b.getF32Type();
  auto indexToF32 = [](OpBuilder &builder, Location loc, Value v) -> Value {
    v = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), v);
    return builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), v);
  };

  // We will need exp(-2 * PI * j / m * I), compute "-2 * PI / m" for imag part
  // first.
  Value coeff = b.create<arith::ConstantFloatOp>(
      loc, llvm::APFloat(static_cast<float>(-2 * acos(-1))), f32Type);
  coeff = b.create<arith::DivFOp>(loc, coeff, indexToF32(b, loc, wholeSize));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{}, operands, maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value lhsReal = args[0];
        Value lhsImag = args[1];
        Value rhsReal = args[2];
        Value rhsImag = args[3];

        // Compute "-2 * PI / m * j"
        Value w = b.create<arith::MulFOp>(
            loc, coeff,
            indexToF32(b, loc, b.create<linalg::IndexOp>(loc, rank - 1)));
        Value wReal = b.create<math::CosOp>(loc, w);
        Value wImag = b.create<math::SinOp>(loc, w);

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

void FftOp::generateScalarImplWithCoeffBuf(OpBuilder &b, Location loc,
                                           ArrayRef<Value> operands) {
  auto rank = getOperandRank();
  SmallVector<AffineMap> maps;
  // The size of coefficent buffer is epxected to match `2^(stage-1)`, which
  // equals to the last dim of operands.
  maps.append(
      2, AffineMap::get(rank, 0, b.getAffineDimExpr(rank - 1), b.getContext()));
  maps.append(operands.size(), b.getMultiDimIdentityMap(rank));

  b.create<linalg::GenericOp>(
      loc, TypeRange{}, ValueRange{getRealCoeff(), getImagCoeff()}, operands,
      maps, getLoopIteratorTypes(),
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value wReal = args[0];
        Value wImag = args[1];
        Value lhsReal = args[2];
        Value lhsImag = args[3];
        Value rhsReal = args[4];
        Value rhsImag = args[5];

        // t = w * a[k + j + mh];
        // ->  (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        Value xu = b.create<arith::MulFOp>(loc, wReal, rhsReal);
        Value yv = b.create<arith::MulFOp>(loc, wImag, rhsImag);
        Value xv = b.create<arith::MulFOp>(loc, wReal, rhsImag);
        Value yu = b.create<arith::MulFOp>(loc, wImag, rhsReal);
        Value tReal = b.create<arith::SubFOp>(loc, xu, yv);
        Value tImag = b.create<arith::AddFOp>(loc, xv, yu);

        // cplx u = a[k + j];
        // a[k + j] = u + t;
        // a[k + j + mh] = u - t;
        Value r1 = b.create<arith::AddFOp>(loc, lhsReal, tReal);
        Value r2 = b.create<arith::AddFOp>(loc, lhsImag, tImag);
        Value r3 = b.create<arith::SubFOp>(loc, lhsReal, tReal);
        Value r4 = b.create<arith::SubFOp>(loc, lhsImag, tImag);
        b.create<linalg::YieldOp>(loc, ValueRange{r1, r2, r3, r4});
      });
}

// Generates FFT stage scalar implementation. This follows Cooley–Tukey FFT
// algorithm. The pseudo reference code is:
//   let s <- stage of linalg_ext.fft
//   int m = 1 << s;
//   int mh = m >> 1;
//   for (int k = 0; k < n; k += m) {
//     for (int j = 0; j < mh; ++j) {
//       cplx w = exp(-2 * PI * j / m * I);
//       cplx t = w * a[k + j + mh];
//       cplx u = a[k + j];
//       a[k + j] = u + t;
//       a[k + j + mh] = u - t;
//     }
//   }
LogicalResult FftOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                  ValueRange ivs) {
  Value real = getReal();
  Value imag = getImag();
  Value stage = getStage();
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value wholeSize = b.create<arith::ShLIOp>(loc, one, stage);
  Value halfSize = b.create<arith::ShRSIOp>(loc, wholeSize, one);

  auto rank = getOperandRank();
  SmallVector<Value> operands;
  SmallVector<OpFoldResult> lhsIvs(ivs.begin(), ivs.end());
  SmallVector<OpFoldResult> ones(rank, b.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes(rank, b.getIndexAttr(1));
  sizes.back() = halfSize;
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, lhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, lhsIvs, sizes, ones));

  SmallVector<OpFoldResult> rhsIvs(ivs.begin(), ivs.end());
  rhsIvs.back() =
      b.create<arith::AddIOp>(loc, ivs.back(), halfSize).getResult();
  operands.push_back(
      b.create<memref::SubViewOp>(loc, real, rhsIvs, sizes, ones));
  operands.push_back(
      b.create<memref::SubViewOp>(loc, imag, rhsIvs, sizes, ones));

  if (hasCoeff()) {
    generateScalarImplWithCoeffBuf(b, loc, operands);
  } else {
    generateScalarImplWithoutCoeffBuf(b, loc, operands, wholeSize);
  }

  return success();
}

FailureOr<TilingResult>
FftOp::getTiledImplementation(OpBuilder &builder,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  SmallVector<Value> tiledOperands(3);
  tiledOperands[0] = getStage();
  tiledOperands[1] = getRealCoeff();
  tiledOperands[2] = getImagCoeff();
  SmallVector<Type, 4> resultTypes;

  for (auto out : getOutputs()) {
    tiledOperands.push_back(
        getSlice(builder, getLoc(), out, offsets, sizes, strides));
    if (hasPureTensorSemantics()) {
      resultTypes.push_back(tiledOperands.back().getType());
    }
  }
  Operation *tiledFftOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledFftOp},
                      SmallVector<Value>(tiledFftOp->getResults())};
}

LogicalResult FftOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

LogicalResult
FftOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

LogicalResult ScanOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = cast<ShapedType>(accumulator().getType());
  auto inputType = cast<ShapedType>(input().getType());
  auto outputType = cast<ShapedType>(output().getType());
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op->emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension()) {
      expectedAccumulatorShape.push_back(inputShapes[i]);
    }
  }
  if (llvm::any_of(llvm::zip_equal(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip_equal(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

SmallVector<Range> ScanOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> ScanOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

// Generates naive scalar implementation of scan for a given operator f.
// For inclusive,
//     output[0] = input[0]
//     output[i] = f(output[i-1], input[i])
//
// For exclusive,
//     output[0] = 0
//     output[i] = f(output[i-1], input[i-1])

LogicalResult ScanOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  SmallVector<Value> indices, scanBlkArgs;
  indices.append(ivs.begin(), ivs.end());
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  auto scanDim = getDimension();
  auto cond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                      indices[scanDim], zero);
  bool isInclusive = getInclusive();
  SmallVector<Value> accIndices;
  for (int i = 0; i < indices.size(); i++) {
    if (i != scanDim) {
      accIndices.push_back(indices[i]);
    }
  }

  auto scfIf = b.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        if (isInclusive) {
          auto value = b.create<memref::LoadOp>(loc, input(), indices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        } else {
          auto value = b.create<memref::LoadOp>(loc, accumulator(), accIndices);
          b.create<memref::StoreOp>(loc, value, output(), indices);
        }
        b.create<scf::YieldOp>(loc);
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> indices(ivs.begin(), ivs.end());
        Value iv = indices[scanDim];
        Value ivMinusOne = b.create<arith::SubIOp>(loc, iv, one);
        indices[scanDim] = ivMinusOne;
        scanBlkArgs.push_back(b.create<memref::LoadOp>(loc, output(), indices));
        Value i0;
        if (!isInclusive)
          i0 = b.create<memref::LoadOp>(loc, input(), indices);
        indices[scanDim] = iv;
        if (isInclusive)
          i0 = b.create<memref::LoadOp>(loc, input(), indices);
        scanBlkArgs.push_back(i0);
      });

  auto &srcBlock = getRegion().front();
  Region &region = scfIf.getElseRegion();
  IRMapping bvm;
  {
    OpBuilder::InsertionGuard guard(b);
    auto &block = region.front();
    b.setInsertionPointToEnd(&block);
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), scanBlkArgs)) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvm);
    }
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        output(), indices);
    b.create<memref::StoreOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)),
        accumulator(), accIndices);
    b.create<scf::YieldOp>(loc);
  }
  return success();
}

FailureOr<TilingResult>
ScanOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        accumOffsets, accumSizes,
                                        accumStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
  }

  Operation *tiledScanOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledScanOp},
                      SmallVector<Value>(tiledScanOp->getResults())};
}

LogicalResult ScanOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

LogicalResult ScanOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult
ScanOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected exactly one input");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected exactly one output");
  }
  auto inputType = cast<ShapedType>(input().getType());
  auto outputType = cast<ShapedType>(output().getType());
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expexted input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip_equal(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return !ShapedType::isDynamic(std::get<0>(s)) &&
                            !ShapedType::isDynamic(std::get<1>(s)) &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }

  int64_t rank = getOperandRank();
  llvm::SmallSetVector<int64_t, 4> s;
  for (auto dim : dims()) {
    if (dim < 0 || dim >= rank) {
      return op->emitOpError("all the dimensions must be within [0, ")
             << rank << ")";
    }
    if (s.contains(dim)) {
      return op->emitOpError("expected dimensions numbers are all unique");
    }
    s.insert(dim);
  }

  return success();
}

SmallVector<utils::IteratorType> ReverseOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

SmallVector<Range> ReverseOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, input(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

LogicalResult ReverseOp::generateScalarImplementation(OpBuilder &b,
                                                      Location loc,
                                                      ValueRange ivs) {
  SmallVector<Value> mirrorIndices(ivs.begin(), ivs.end());
  for (auto dim : dims()) {
    auto size = getDimValue(b, loc, input(), dim);
    size = b.create<arith::SubIOp>(loc, size,
                                   b.create<arith::ConstantIndexOp>(loc, 1));
    mirrorIndices[dim] = b.create<arith::SubIOp>(loc, size, mirrorIndices[dim]);
  }
  Value val = b.create<memref::LoadOp>(loc, input(), ivs);
  b.create<memref::StoreOp>(loc, val, output(), mirrorIndices);
  return success();
}

FailureOr<TilingResult>
ReverseOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();
  SmallVector<OpFoldResult> mirrorOffsets, mirrorSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, mirrorOffsets,
                                   mirrorSizes))) {
    return {};
  }

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, input(), offsets, sizes, strides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, output(), mirrorOffsets, sizes, strides));
    resultTypes.push_back(tiledOperands[1].getType());
  } else {
    tiledOperands.emplace_back(
        getSlice(builder, loc, output(), mirrorOffsets, sizes, strides));
  }

  Operation *tiledRevOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledRevOp},
                      SmallVector<Value>(tiledRevOp->getResults())};
}

LogicalResult ReverseOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  AffineMap map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/3, {sym0 - sym1 - sym2});
  resultOffsets.assign(offsets.begin(), offsets.end());
  Location loc = getLoc();
  for (auto dim : dims()) {
    Value size = getDimValue(builder, loc, input(), dim);
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, resultOffsets[dim]);
    Value tileSize = getValueOrCreateConstantIndexOp(builder, loc, sizes[dim]);
    resultOffsets[dim] = builder
                             .create<affine::AffineApplyOp>(
                                 loc, map, ValueRange{size, offset, tileSize})
                             .getResult();
  }
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

LogicalResult
ReverseOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

LogicalResult TopkOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1 && getNumDpsInputs() != 2) {
    return op->emitOpError("expected one or two input operands");
  }
  if (getNumDpsInits() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (getDimension() >= getInputRank()) {
    return op->emitOpError("dimension exceeds rank");
  }
  // Ensure input/output element types match
  auto inputValuesType = cast<ShapedType>(values().getType());
  auto outputValuesType = cast<ShapedType>(outputValues().getType());
  if (inputValuesType.getElementType() != outputValuesType.getElementType()) {
    return op->emitOpError("expected input/output value types to be identical");
  }
  // Indices must be int if provided
  auto outputIndicesType = cast<ShapedType>(outputIndices().getType());
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (!inputIndicesType.getElementType().isInteger(32) ||
        !outputIndicesType.getElementType().isInteger(32)) {
      return op->emitOpError("expected input/output indices types to be int32");
    }
  }

  // Ranks must match
  if (inputValuesType.getRank() != outputValuesType.getRank()) {
    return op->emitOpError("expected input/output to have the same rank");
  }
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (inputIndicesType.getRank() != outputIndicesType.getRank()) {
      return op->emitOpError("expected input/output to have the same rank");
    }
  }
  // Input indicies and values must have the same shape.
  if (auto inputIndices = indices()) {
    auto inputIndicesType = cast<ShapedType>(inputIndices->getType());
    if (failed(verifyCompatibleShape(inputValuesType, inputIndicesType))) {
      return op->emitOpError("input indices/values shape must match");
    }
  }
  // Output indicies and values must have the same shape.
  if (failed(verifyCompatibleShape(outputValuesType, outputIndicesType))) {
    return op->emitOpError("output indices/values shape must match");
  }
  // Input shape must match the output shape except for the dimension()
  uint64_t dim = getDimension();
  if (!llvm::all_of(
          llvm::enumerate(llvm::zip_equal(inputValuesType.getShape(),
                                          outputValuesType.getShape())),
          [dim](auto e) {
            if (e.index() == dim) {
              return true;
            }
            std::tuple<int64_t, int64_t> s = e.value();
            return succeeded(
                verifyCompatibleShape(std::get<0>(s), std::get<1>(s)));
          })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  // Check region compatibility
  Block &block = getRegion().front();
  if (block.getNumArguments() != 2) {
    return op->emitOpError("region block should have 2 arguments");
  }
  if (block.getArgument(0).getType() != inputValuesType.getElementType() ||
      block.getArgument(1).getType() != inputValuesType.getElementType()) {
    return op->emitOpError("region block types must match input");
  }
  auto terminatorOp = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!terminatorOp || !terminatorOp.getOperand(0).getType().isInteger(1)) {
    return op->emitOpError("region block must end with a linalg_ext.yield i1!");
  }
  return success();
}

SmallVector<Range> TopkOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = values();
  for (auto [idx, val] : llvm::enumerate(getInputType().getShape())) {
    loopBounds[idx].offset = zero;
    loopBounds[idx].size = getDimValue(builder, loc, source, idx);
    loopBounds[idx].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> TopkOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

LogicalResult TopkOp::generateScalarImplementation(OpBuilder &b, Location loc,
                                                   ValueRange ivs) {
  uint64_t kDim = getDimension();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value initialValue = b.create<memref::LoadOp>(loc, values(), ivs);

  // If the indices tensor is not provided, the value index is derived from the
  // loop induction variables.
  Value initialIndex;
  if (indices()) {
    initialIndex = b.create<memref::LoadOp>(loc, *indices(), ivs);
  } else {
    Value rawInitialIndex = ivs[kDim];
    initialIndex =
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), rawInitialIndex);
  }

  // Compute K (ub) from the selected dim of the output
  Value ub = b.create<memref::DimOp>(loc, outputValues(), getDimension());

  // Inner K loop functions:
  //   Load current K value and index
  //   Compare N/K using inserted block compare
  //   Check if N == K using strict weak ordering, select which index came first
  //   Select new K value from N/K comparison
  //   Select new K index from N/K comparison or which index came first
  //   Store new k value and index
  //   Yield loop carry values after K selection
  Value kValue, kIndex;
  auto scfFor = b.create<scf::ForOp>(
      loc, zero, ub, one, ValueRange{initialValue, initialIndex},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange loopCarryValues) {
        SmallVector<Value> indices(ivs);
        indices[kDim] = iv;
        kValue = b.create<memref::LoadOp>(loc, outputValues(), indices);
        kIndex = b.create<memref::LoadOp>(loc, outputIndices(), indices);
      });

  SmallVector<Value> indices(ivs);
  indices[kDim] = scfFor.getInductionVar();
  auto loopCarryValues = scfFor.getRegionIterArgs();

  // Retrieve region as black box comparision function f(x,y). Plug into op.
  auto &srcBlock = getRegion().front();
  IRMapping bvmF; // f(x,y)
  IRMapping bvmR; // f(y,x)
  {
    // Save previous insertion point. Continue within loop body.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&scfFor.getRegion().front());
    SmallVector<Value> forwardValues{loopCarryValues[0], kValue};
    SmallVector<Value> reverseValues{kValue, loopCarryValues[0]};
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), forwardValues)) {
      bvmF.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto it : llvm::zip_equal(srcBlock.getArguments(), reverseValues)) {
      bvmR.map(std::get<0>(it), std::get<1>(it));
    }
    for (auto &blockOp : srcBlock.without_terminator()) {
      b.clone(blockOp, bvmF);
      b.clone(blockOp, bvmR);
    }
    Value forwardCmpRes = bvmF.lookup(srcBlock.getTerminator()->getOperand(0));
    Value reverseCmpRes = bvmR.lookup(srcBlock.getTerminator()->getOperand(0));

    // Check value equality using strictly weak ordering from the region:
    //   f(x,y) --> forwardCmpRes
    //   f(y,x) --> reverseCmpRes
    //   if forwardCmpRes == reverseCmpRes then select which came first
    Value cmpValuesEqual = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, forwardCmpRes, reverseCmpRes);
    Value cmpFirstIndex = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, loopCarryValues[1], kIndex);
    Value combinedCmpEqRes =
        b.create<arith::AndIOp>(loc, cmpValuesEqual, cmpFirstIndex);
    // True if N > K or N came before K
    Value indexCmpRes =
        b.create<arith::OrIOp>(loc, forwardCmpRes, combinedCmpEqRes);
    // Select results for K based on comparisons
    Value resultKValue = b.create<arith::SelectOp>(loc, forwardCmpRes,
                                                   loopCarryValues[0], kValue);
    Value resultKIndex =
        b.create<arith::SelectOp>(loc, indexCmpRes, loopCarryValues[1], kIndex);
    b.create<memref::StoreOp>(loc, resultKValue, outputValues(), indices);
    b.create<memref::StoreOp>(loc, resultKIndex, outputIndices(), indices);
    // Select loop carry, opposite of K results
    Value resultCarryValue = b.create<arith::SelectOp>(
        loc, forwardCmpRes, kValue, loopCarryValues[0]);
    Value resultCarryIndex =
        b.create<arith::SelectOp>(loc, indexCmpRes, kIndex, loopCarryValues[1]);
    b.create<scf::YieldOp>(loc, ValueRange{resultCarryValue, resultCarryIndex});
  }
  return success();
}

FailureOr<TilingResult>
TopkOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();

  SmallVector<OpFoldResult> outputOffsets, outputSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, outputOffsets,
                                   outputSizes))) {
    return {};
  }

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, values(), offsets, sizes, strides));
  if (indices()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, *indices(), offsets, sizes, strides));
  }

  // Replace the tile size for the K dimension to use the output size instead of
  // the input size.
  Value kSize = getDimValue(builder, getLoc(), outputValues(), getDimension());
  outputSizes[getDimension()] = getAsOpFoldResult(kSize);

  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[0], offsets, outputSizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[1], offsets, outputSizes, strides));
  SmallVector<Type, 2> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 2].getType());
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 1].getType());
  }

  Operation *tiledTopkOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return TilingResult{{tiledTopkOp},
                      SmallVector<Value>(tiledTopkOp->getResults())};
}

LogicalResult TopkOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  Value kSize = getDimValue(builder, getLoc(), getDpsInits()[resultNumber],
                            getDimension());
  resultSizes[getDimension()] = getAsOpFoldResult(kSize);
  return success();
}

LogicalResult
TopkOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

/// Return true if at least one element in `tiles` is zero.
static bool hasZeros(ArrayRef<OpFoldResult> tiles) {
  return llvm::any_of(
      tiles, [&](OpFoldResult tile) { return isConstantIntValue(tile, 0); });
}

/// Check if we have enough static information to catch undefined behavior when
/// the tile size does not divide perfectly the dimension of the input tensor.
static bool
areNotFullTiles(ArrayRef<int64_t> inputShape,
                DenseMap<int64_t, OpFoldResult> const &dimAndTileMapping) {
  int64_t rank = inputShape.size();
  for (int64_t dim = 0; dim < rank; dim++) {
    if (ShapedType::isDynamic(inputShape[dim]))
      continue;
    auto it = dimAndTileMapping.find(dim);
    if (it != dimAndTileMapping.end()) {
      std::optional<int64_t> constantTile = getConstantIntValue(it->second);
      if (!constantTile)
        continue;
      if (inputShape[dim] % (*constantTile) != 0)
        return true;
    }
  }
  return false;
}

/// Utility function shared between Pack and UnPack to get the tile sizes as
/// OpFoldResults.
// TODO: interface or base class in .td
template <typename OpTy>
static SmallVector<OpFoldResult> getMixedTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<OpFoldResult> mixedInnerTiles;
  unsigned dynamicValIndex = 0;
  OpBuilder b(op.getContext());
  for (int64_t tileSize : op.getStaticInnerTiles()) {
    if (!ShapedType::isDynamic(tileSize)) {
      mixedInnerTiles.push_back(b.getIndexAttr(tileSize));
    } else {
      mixedInnerTiles.push_back(op.getInnerTiles()[dynamicValIndex++]);
    }
  }
  return mixedInnerTiles;
}

/// Return the tile sizes as `int64_t`. If a tile size is dynamic a sentinel
/// `kDynamic` is introduced at that position in the returned vector.
template <typename OpTy>
static SmallVector<int64_t> getStaticTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(op.getMixedTiles(), dynamicTiles, staticTiles);
  return staticTiles;
}

/// Utility function shared between Pack and UnPack to get a map between
/// `dim_pos` and `inner_tiles`.
// TODO: interface or base class in .td
template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMapping(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  ArrayRef<int64_t> dimsToBlock = op.getInnerDimsPos();
  SmallVector<OpFoldResult> tiles = op.getMixedTiles();
  assert(tiles.size() == dimsToBlock.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension with the tile factor.
  for (auto i : llvm::seq<int64_t>(0, dimsToBlock.size())) {
    dimAndTileMapping[dimsToBlock[i]] = tiles[i];
  }
  return dimAndTileMapping;
}

/// Utility function to build the iteration domain for `packOp` or `unPackOp`.
template <typename OpTy>
static SmallVector<Range> getIterationDomain(OpTy op, OpBuilder &builder) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  OpBuilder::InsertionGuard g(builder);
  Location loc = op.getLoc();
  int64_t rank = (std::is_same<OpTy, PackOp>::value) ? op.getInputRank()
                                                     : op.getOutputRank();
  SmallVector<Range> loopBounds(rank);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultShape;
  (void)op.reifyResultShapes(builder, resultShape);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].stride = one;
    loopBounds[dim].size = resultShape[0][dim];
  }
  return loopBounds;
}

/// Common verifier for `PackOp` and `UnPackOp`.
template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Operation *op = packOrUnPack.getOperation();
  ShapedType unpackedType = (std::is_same<OpTy, PackOp>::value)
                                ? packOrUnPack.getInputType()
                                : packOrUnPack.getOutputType();
  int64_t unpackedRank = unpackedType.getRank();
  ArrayRef<int64_t> innerDimsPos = packOrUnPack.getInnerDimsPos();
  ArrayRef<int64_t> outerDimPerm = packOrUnPack.getOuterDimsPerm();
  // Verify tiles. Make sure each provided tile is non-zero.
  SmallVector<OpFoldResult> mixedTiles = packOrUnPack.getMixedTiles();
  if (hasZeros(mixedTiles)) {
    return op->emitError("invalid tile factor");
  }
  if (isInvalid(innerDimsPos, unpackedRank)) {
    return op->emitError("invalid inner_dims_pos vector");
  }
  if (isInvalid(outerDimPerm, unpackedRank)) {
    return op->emitError("invalid outer_dims_perm vector");
  }
  if (mixedTiles.size() != innerDimsPos.size()) {
    return op->emitError(
        "blocking factors must equal the number of dimensions to block");
  }

  // Blocking factors must be less or equal than the input rank, and must
  // match the number of `dims_pos`.
  if (mixedTiles.size() > unpackedRank) {
    return op->emitError(
        "blocking factors must be less or equal than the input rank");
  }

  ShapedType packedType = (std::is_same<OpTy, PackOp>::value)
                              ? packOrUnPack.getOutputType()
                              : packOrUnPack.getInputType();
  int64_t packedRank = packedType.getRank();
  // Require output rank to match input rank + number of blocking factors.
  if (unpackedRank + mixedTiles.size() != packedRank) {
    return op->emitError(
        "packed rank must equal unpacked rank + blocking factors");
  }

  // Verify result shape is greater than the minimum expected
  // by the pack operation, and that the output shape
  // represents full tiles.
  ShapedType expectedPackedType = PackOp::getPackedType(
      unpackedType, packOrUnPack.getStaticTiles(), innerDimsPos, outerDimPerm);
  if (!isSmallerThan(expectedPackedType.getShape(), packedType.getShape())) {
    return op->emitError("the shape of output is not large enough to hold the "
                         "packed data. Expected at least ")
           << expectedPackedType << ", got " << packedType;
  }
  if (!llvm::all_of(
          llvm::zip_equal(packedType.getShape().take_back(mixedTiles.size()),
                          mixedTiles),
          [](std::tuple<int64_t, OpFoldResult> it) {
            std::optional<int64_t> constTileSize =
                getConstantIntValue(std::get<1>(it));
            int64_t shape = std::get<0>(it);
            if (!constTileSize) {
              // If specified tile size is dynamic, output shape should
              // be dynamic too.
              return ShapedType::isDynamic(shape);
            } else {
              if (ShapedType::isDynamic(shape)) {
                // For the shape being dynamic when tile size is
                // specified, return true. In canonical form a constant
                // tile size should lead to constant shape of the tiled
                // dimension, but not needed for verification.
                return true;
              }
              return shape == constTileSize.value();
            }
          })) {
    return op->emitError("mismatch in inner tile sizes specified and shaped of "
                         "tiled dimension in the packed type");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for pack ops.
void PackOp::build(OpBuilder &builder, OperationState &state, Value source,
                   Value output, ArrayRef<int64_t> innerDimsPos,
                   ArrayRef<OpFoldResult> innerTiles,
                   std::optional<Value> paddingValue,
                   ArrayRef<int64_t> outerDimsPerm) {
  assert(innerDimsPos.size() == innerTiles.size() &&
         "number of tile sizes specified must match the specified number of "
         "original dimensions to be tiled");
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (outputType.isa<RankedTensorType>()) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes),
        (paddingValue ? paddingValue.value() : nullptr));
}

LogicalResult PackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }

  // Bail out if the tile does not divide the dimension fully. In the case of
  // dynamic tile factors or dimensions, having a partial tile is undefined
  // behavior.
  auto dimAndTileMapping = getDimAndTileMapping();
  if (!getPaddingValue() &&
      areNotFullTiles(getInputShape(), dimAndTileMapping)) {
    return emitOpError("invalid tile factor provided. Only full tiles are "
                       "supported when padding_value is not set");
  }

  if (auto paddingValue = getPaddingValue()) {
    if (paddingValue.getType() != getInputType().getElementType()) {
      return emitOpError("expected padding_value has ")
             << getInputType().getElementType()
             << " but got: " << paddingValue.getType();
    }
  }
  return success();
}

SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> PackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

// Helper for PackOp::{getResultShape,getPackedType}. Returns the shape of the
// packed type. Having a shared helper helps implement these two methods in a
// way that ensures that they agree on which dimensions are dynamic.
static SmallVector<int64_t> getPackOpResultTypeShape(
    ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> innerTileSizes,
    ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultShape = llvm::to_vector(sourceShape);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    if (ShapedType::isDynamic(resultShape[tiledDim])) {
      continue;
    }
    if (ShapedType::isDynamic(innerTileSizes[idx])) {
      resultShape[tiledDim] = ShapedType::kDynamic;
      continue;
    }
    resultShape[tiledDim] = ceilDiv(resultShape[tiledDim], innerTileSizes[idx]);
  }

  // Swap tile loops if outer_dims_perm is available.
  resultShape = interchange<int64_t>(resultShape, outerDimsPerm, /*offset=*/0);

  // Append the inner tile dimensions.
  resultShape.append(innerTileSizes.begin(), innerTileSizes.end());
  return resultShape;
}

SmallVector<OpFoldResult> PackOp::getResultShape(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> sourceDims,
    ArrayRef<OpFoldResult> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<OpFoldResult> resultDims = llvm::to_vector(sourceDims);

  AffineExpr s0, s1;
  bindSymbols(builder.getContext(), s0, s1);
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  for (auto [idx, tiledDim] : llvm::enumerate(innerDimsPos)) {
    resultDims[tiledDim] = affine::makeComposedFoldedAffineApply(
        builder, loc, ceilDivExpr, {resultDims[tiledDim], innerTileSizes[idx]});
  }
  if (!outerDimsPerm.empty()) {
    resultDims =
        interchange<OpFoldResult>(resultDims, outerDimsPerm, /*offset=*/0);
  }
  resultDims.append(innerTileSizes.begin(), innerTileSizes.end());

  SmallVector<int64_t> resultTypeShape =
      getPackOpResultTypeShape(asShapeWithAnyValueAsDynamic(sourceDims),
                               asShapeWithAnyValueAsDynamic(innerTileSizes),
                               innerDimsPos, outerDimsPerm);

  // Fix-up `resultDims` to ensure that they are Value's if and only if the
  // result type shape says it's a dynamic dim. This is needed as callers may
  // use dispatchIndexOpFoldResults on the result, and rely on exact number of
  // dynamic dims returned by that.
  for (unsigned i = 0; i < resultDims.size(); ++i) {
    if (!ShapedType::isDynamic(resultTypeShape[i])) {
      continue;
    }
    resultDims[i] =
        getValueOrCreateConstantIndexOp(builder, loc, resultDims[i]);
  }

  return resultDims;
}

SmallVector<OpFoldResult> PackOp::getResultShape(OpBuilder &builder) {
  return tensor::getMixedSizes(builder, getLoc(), getOutput());
}

ShapedType PackOp::getPackedType(ShapedType sourceType,
                                 ArrayRef<int64_t> innerTileSizes,
                                 ArrayRef<int64_t> innerDimsPos,
                                 ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> resultTypeShape = getPackOpResultTypeShape(
      sourceType.getShape(), innerTileSizes, innerDimsPos, outerDimsPerm);

  return TypeSwitch<ShapedType, ShapedType>(sourceType)
      .Case<RankedTensorType>([&](auto shapedType) {
        return RankedTensorType::get(resultTypeShape,
                                     shapedType.getElementType());
      })
      .Case<MemRefType>([&](auto shapedType) {
        return MemRefType::get(resultTypeShape, shapedType.getElementType());
      })
      .Default([&](Type t) {
        assert(false && "unexpected type");
        return nullptr;
      });
}

DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

SmallVector<Range> PackOp::getIterationDomain(OpBuilder &builder) {
  return LinalgExt::getIterationDomain(*this, builder);
}

/// Generate the body of the innermost loop of the scalar implementation
/// of `pack` operation.
static void generatePackOpScalarImplementationBody(PackOp packOp,
                                                   OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  // Note: `ivs` are already in the correct order, possibly interchanged based
  // on `dims_pos`. However, connecting the loops with the access patterns is
  // difficult - What is the relation between the position of the tile loop and
  // the point loop? However, if we interchange `ivs` once more to go to the
  // canonical blocking format: ABCabc, this connection becomes trivial: Each
  // point loop is pointLoopsOffset + inputRank away from the tiled loop.
  ArrayRef<int64_t> dimsToInnerBlock = packOp.getInnerDimsPos();
  ArrayRef<int64_t> dimsToOuterBlock = packOp.getOuterDimsPerm();

  SmallVector<Value> interchangedIvs = ivs;
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(dimsToInnerBlock, packOp.getInputRank());
  interchangedIvs = interchange<Value>(interchangedIvs, interchangeVector,
                                       /*offset=*/packOp.getInputRank());
  if (!dimsToOuterBlock.empty()) {
    interchangeVector =
        computeInterchangeFromDimPos(dimsToOuterBlock, packOp.getInputRank());
    interchangedIvs =
        interchange<Value>(interchangedIvs, interchangeVector, /*offset=*/0);
  }

  SmallVector<OpFoldResult> tiles = packOp.getMixedTiles();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  SmallVector<OpFoldResult> sourceIndices;
  size_t pointLoopsOffset = 0;
  int64_t inputRank = packOp.getInputRank();
  for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
    if (dimAndTileMapping.count(dim)) {
      AffineExpr i, j, tile;
      bindDims(builder.getContext(), i, j);
      bindSymbols(builder.getContext(), tile);
      OpFoldResult sourceIndex = affine::makeComposedFoldedAffineApply(
          builder, loc, i * tile + j,
          ArrayRef<OpFoldResult>{
              interchangedIvs[dim],
              interchangedIvs[pointLoopsOffset + packOp.getInputRank()],
              dimAndTileMapping[dim]});
      sourceIndices.push_back(sourceIndex);
      ++pointLoopsOffset;
    } else {
      sourceIndices.push_back(interchangedIvs[dim]);
    }
  }

  auto createLoad = [&]() -> Value {
    return builder.create<memref::LoadOp>(
        loc, packOp.getInput(),
        getValueOrCreateConstantIndexOp(builder, loc, sourceIndices));
  };
  Value scalar;
  if (auto paddingValue = packOp.getPaddingValue()) {
    ArithBuilder arithBuilder(builder, loc);
    Value isInBounds;
    for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
      Value idx =
          getValueOrCreateConstantIndexOp(builder, loc, sourceIndices[dim]);
      Value cond = arithBuilder.slt(
          idx, getDimValue(builder, loc, packOp.getInput(), dim));
      isInBounds = dim == 0 ? cond : arithBuilder._and(isInBounds, cond);
    }
    scalar = builder
                 .create<scf::IfOp>(
                     loc, isInBounds, /*thenBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, createLoad());
                     },
                     /*elseBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, paddingValue);
                     })
                 .getResult(0);
  } else {
    scalar = createLoad();
  }

  builder.create<memref::StoreOp>(loc, scalar, packOp.getOutput(), ivs);
}

LogicalResult PackOp::generateScalarImplementation(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  OpBuilder::InsertionGuard g(builder);
  // The `ivs` already represent the position into the output tensor for the
  // non data-tile dimensions.
  SmallVector<Value> ivVec = llvm::to_vector(ivs);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape))) {
    return getOperation()->emitOpError("failed to reify result shape");
  }
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  // Generate the loops that iterate over the data tile.
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

  // All loops except the innermost are simple loops that just iterate
  // over the tile dimensions.
  for (auto dataTileDim :
       llvm::seq<unsigned>(getInputRank(), getOutputRank() - 1)) {
    Value ub = getValueOrCreateConstantIndexOp(builder, loc,
                                               outputShape[0][dataTileDim]);
    scf::ForOp loop = builder.create<scf::ForOp>(loc, zero, ub, one);
    builder.setInsertionPointToStart(loop.getBody());
    ivVec.push_back(loop.getInductionVar());
  }
  // The body of the innermost loops does the actual data movement.
  builder.create<scf::ForOp>(
      loc, zero,
      getValueOrCreateConstantIndexOp(builder, loc, outputShape[0].back()), one,
      ValueRange{},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, Value iv,
          ValueRange regionIterArgs) {
        ivVec.push_back(iv);
        generatePackOpScalarImplementationBody(*this, bodyBuilder, bodyLoc,
                                               ivVec);
        bodyBuilder.create<scf::YieldOp>(bodyLoc);
      });
  return success();
}

LogicalResult
PackOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

/// Custom builder methods for unpack ops.
void UnPackOp::build(OpBuilder &builder, OperationState &state, Value source,
                     Value output, ArrayRef<int64_t> innerDimsPos,
                     ArrayRef<OpFoldResult> innerTiles,
                     ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(innerTiles, dynamicTileSizes, staticTileSizes);
  SmallVector<Type> resultType;
  auto outputType = output.getType();
  if (outputType.isa<RankedTensorType>()) {
    resultType.push_back(outputType);
  }
  build(builder, state, resultType, source, output,
        outerDimsPerm.empty() ? nullptr
                              : builder.getDenseI64ArrayAttr(outerDimsPerm),
        builder.getDenseI64ArrayAttr(innerDimsPos), dynamicTileSizes,
        builder.getDenseI64ArrayAttr(staticTileSizes));
}

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return LinalgExt::getMixedTiles(*this);
}

SmallVector<int64_t> UnPackOp::getStaticTiles() {
  return LinalgExt::getStaticTiles(*this);
}

DenseMap<int64_t, OpFoldResult> UnPackOp::getDimAndTileMapping() {
  return LinalgExt::getDimAndTileMapping(*this);
}

LogicalResult UnPackOp::generateScalarImplementation(OpBuilder &builder,
                                                     Location loc,
                                                     ValueRange ivs) {
  assert(ivs.size() == getOutputRank() &&
         "number of ivs must match the rank of the output tensor");
  OpBuilder::InsertionGuard g(builder);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape))) {
    return getOperation()->emitOpError("failed to reify result shape");
  }
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  DenseMap<int64_t, OpFoldResult> dimAndTileMapping = getDimAndTileMapping();
  // untiled loops and tile loops induction variables.
  SmallVector<Value> inputIvs;
  // point loops induction variables.
  SmallVector<Value> inputIvsPointLoops;
  inputIvs.reserve(getOutputRank());
  inputIvsPointLoops.reserve(dimAndTileMapping.size());
  for (auto dim : llvm::seq<int64_t>(0, getOutputRank())) {
    if (dimAndTileMapping.count(dim)) {
      affine::DivModValue divMod =
          affine::getDivMod(builder, loc, ivs[dim],
                            getValueOrCreateConstantIndexOp(
                                builder, loc, dimAndTileMapping[dim]));
      inputIvsPointLoops.push_back(divMod.remainder);
      inputIvs.push_back(divMod.quotient);
    } else {
      inputIvs.push_back(ivs[dim]);
    }
  }

  // TODO: (lorenzo) simplify the logic a bit. There is `ivs`,
  // `inputIvsPointLoops` and `inputIvs`.
  assert(inputIvsPointLoops.size() + inputIvs.size() == getInputRank() &&
         "expect same number of iduction variables equals to input rank");
  // interchange the point loops induction variables based on `inner_dim_pos`.
  ArrayRef<int64_t> innerDims = getInnerDimsPos();
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(innerDims, getOutputRank());
  SmallVector<Value> interchangedInputIvsPointLoops = inputIvsPointLoops;
  interchangedInputIvsPointLoops = interchange<Value>(
      interchangedInputIvsPointLoops, interchangeVector, /*offset=*/0);
  // interchange the tiled loops induction variables based on `outer_dims_perm`.
  ArrayRef<int64_t> outerDims = getOuterDimsPerm();
  if (!outerDims.empty()) {
    inputIvs = interchange<Value>(inputIvs, outerDims, /*offset=*/0);
  }

  llvm::append_range(inputIvs, interchangedInputIvsPointLoops);
  Value scalar = builder.create<memref::LoadOp>(loc, getInput(), inputIvs);
  builder.create<memref::StoreOp>(loc, scalar, getOutput(), ivs);
  return success();
}

LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(builder, reifiedReturnShapes);
}

SmallVector<Range> UnPackOp::getIterationDomain(OpBuilder &builder) {
  return LinalgExt::getIterationDomain(*this, builder);
}

LogicalResult UnPackOp::verify() {
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WinogradInputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradInputTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    if ((!inputType.isDynamicDim(0) &&
         inputType.getDimSize(0) > getInputTileSize()) ||
        (inputType.isDynamicDim(1) &&
         inputType.getDimSize(1) > getInputTileSize())) {
      return op->emitOpError("expected input dims not greater than input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError(
          "expected output dims equal to inputTileSize if input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank() + 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank + 2");
  }
  const SmallVector<int64_t> imageDims = imageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  const int64_t outputTileSize = getOutputTileSize();
  const int64_t kernelSize = getKernelSize();
  const int64_t inputTileSize = getInputTileSize();
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), inputTileSize);
  int outputIndex;
  ArrayRef<int64_t> inputShape = inputType.getShape();
  for (int i = 0; i < inputShape.size(); i++) {
    outputIndex = i + numImageDims;
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(i)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] =
          std::ceil((float)(inputShape[i] - kernelSize + 1) / outputTileSize);
    }
  }
  if (isNchw()) {
    permute<Permutation::TTNCHW_TO_TTNHWC>(expectedOutputShape);
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

SmallVector<Range>
WinogradInputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value dest = output();
  SmallVector<Range> loopBounds(getIterationDomainRank());
  int count = 0;
  for (auto dim :
       llvm::seq<int64_t>(imageDimensions().size(), getOutputRank())) {
    loopBounds[count].offset = zero;
    loopBounds[count].size = getDimValue(builder, loc, dest, dim);
    loopBounds[count].stride = one;
    count++;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradInputTransformOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult>
WinogradInputTransformOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  const int cDim = channelDim();

  assert(offsets.size() == 4);
  SmallVector<OpFoldResult> inputOffsets(getInputRank(), zero);
  SmallVector<OpFoldResult> outputOffsets(getOutputRank(), zero);
  const auto hDim = getImageDimensions()[0];
  const auto wDim = getImageDimensions()[1];
  outputOffsets[2] = inputOffsets[0] = offsets[0];
  outputOffsets[3] = offsets[1];
  outputOffsets[4] = offsets[2];
  outputOffsets[5] = inputOffsets[cDim] = offsets[3];

  SmallVector<OpFoldResult> inputStrides(getInputRank(), one);
  SmallVector<OpFoldResult> outputStrides(getOutputRank(), one);
  ReifiedRankedShapedTypeDims reifiedResultShapes, reifiedInputShapes;
  if (failed(reifyResultShapes(builder, reifiedResultShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> outputSizes = reifiedResultShapes[0];
  if (failed(getStaticOrReifiedInputDims(builder, loc, input(),
                                         reifiedInputShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> inputSizes = reifiedInputShapes[0];

  assert(sizes.size() == 4);
  outputSizes[2] = inputSizes[0] = sizes[0];
  outputSizes[3] = sizes[1];
  outputSizes[4] = sizes[2];
  outputSizes[5] = inputSizes[cDim] = sizes[3];

  auto hSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizes[1], offsets[1], inputSizes[hDim], getOutputTileSize(),
      getInputTileSize());
  auto wSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizes[2], offsets[2], inputSizes[wDim], getOutputTileSize(),
      getInputTileSize());

  inputSizes[hDim] = hSizeAndOffset.first;
  inputSizes[wDim] = wSizeAndOffset.first;
  inputOffsets[hDim] = hSizeAndOffset.second;
  inputOffsets[wDim] = wSizeAndOffset.second;

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, input(), inputOffsets, inputSizes, inputStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, output(), outputOffsets,
                                      outputSizes, outputStrides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult WinogradInputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    auto resultShape = cast<ShapedType>(output().getType()).getShape();
    resultSizes = getAsOpFoldResult(builder.getIndexArrayAttr(resultShape));
    resultOffsets =
        SmallVector<OpFoldResult>(getOutputRank(), builder.getIndexAttr(0));
    resultOffsets[2] = offsets[0];
    resultOffsets[3] = offsets[1];
    resultOffsets[4] = offsets[2];
    resultOffsets[5] = offsets[3];
    resultSizes[2] = sizes[0];
    resultSizes[3] = sizes[1];
    resultSizes[4] = sizes[2];
    resultSizes[5] = sizes[3];
    return success();
  }
  return failure();
}

LogicalResult WinogradInputTransformOp::fold(FoldAdaptor,
                                             SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradInputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradFilterTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradFilterTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 4) {
    return op->emitOpError("expected input operand to have rank either 2 or 4");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getKernelSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to kernel size "
                             "if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getInputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to input tile size if "
                             "input is of rank 2");
    }
    return success();
  }

  if (getOutputRank() != getInputRank()) {
    return op->emitOpError("expected output rank to be equal to input rank");
  }
  const ArrayRef<int64_t> kernelDims = getKernelDimensions();
  if (kernelDims.size() != 2) {
    return op->emitOpError("expected only 2 kernel dimensions");
  }
  if (!isHwcf() && !isFchw()) {
    return op->emitOpError(
        "expect kernel dimensions to be either [0, 1] or [2, 3]");
  }
  const int64_t kernelSize = getKernelSize();
  for (auto kernelDim : kernelDims) {
    if (inputType.getDimSize(kernelDim) != kernelSize) {
      return op->emitOpError(
          "expect all kernel dimensions to have the kernel size");
    }
  }
  const int64_t inputTileSize = getInputTileSize();
  SmallVector<int64_t> expectedOutputShape(kernelDims.size(), inputTileSize);
  llvm::SmallSetVector<int64_t, 2> kernelDimsSet(kernelDims.begin(),
                                                 kernelDims.end());
  for (int i = 0; i < inputType.getRank(); i++) {
    if (!kernelDimsSet.contains(i)) {
      expectedOutputShape.push_back(inputType.getDimSize(i));
    }
  }
  if (isFchw()) {
    permute<Permutation::TTFC_TO_TTCF>(expectedOutputShape);
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

LogicalResult WinogradFilterTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradFilterTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WinogradOutputTransformOp
//===----------------------------------------------------------------------===//

LogicalResult WinogradOutputTransformOp::verify() {
  Operation *op = getOperation();
  if (getNumDpsInputs() != 1) {
    return op->emitOpError("expected one input operand");
  }
  if (getNumDpsInits() != 1) {
    return op->emitOpError("expected one output operand");
  }
  auto inputType = getInputType();
  auto outputType = getOutputType();
  unsigned inputRank = inputType.getRank();
  unsigned outputRank = outputType.getRank();

  if (inputRank != 2 && inputRank != 6) {
    return op->emitOpError("expected input operand to have rank either 2 or 6");
  }

  if (inputRank == 2) {
    if (outputRank != 2) {
      return op->emitOpError(
          "expected output operand to have rank 2 if input is of rank 2");
    }
    SmallVector<int64_t> expectedInputShape(2, getInputTileSize());
    if (failed(
            verifyCompatibleShape(expectedInputShape, inputType.getShape()))) {
      return op->emitOpError("expected input dims to be equal to input tile "
                             "size if input is of rank 2");
    }
    SmallVector<int64_t> expectedOutputShape(2, getOutputTileSize());
    if (failed(verifyCompatibleShape(expectedOutputShape,
                                     outputType.getShape()))) {
      return op->emitOpError("expected output dims equal to output tile size "
                             "if input is of rank 2");
    }
    return success();
  }
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (outputRank != inputRank - 2) {
    return op->emitOpError(
        "expected output rank to be equal to input rank - 2");
  }
  const SmallVector<int64_t> imageDims = imageDimensions();
  const size_t numImageDims = imageDims.size();
  llvm::SmallSetVector<int64_t, 2> imageDimsSet(imageDims.begin(),
                                                imageDims.end());
  if (imageDims.size() != 2) {
    return op->emitOpError("expected only 2 image dimensions");
  }
  if (!isNchw() && !isNhwc()) {
    return op->emitOpError(
        "expect image dimensions to be either [1, 2] or [2, 3]");
  }
  SmallVector<int64_t> inputShape(inputType.getShape());
  if (isNchw()) {
    permute<Permutation::TTNHWC_TO_TTNCHW>(inputShape);
  }
  const int64_t outputTileSize = getOutputTileSize();
  SmallVector<int64_t> expectedOutputShape(getOutputRank(), 1);
  int outputIndex;
  for (int i = numImageDims; i < inputShape.size(); i++) {
    outputIndex = i - numImageDims;
    if (ShapedType::isDynamic(inputShape[i])) {
      expectedOutputShape[outputIndex] = inputShape[i];
      continue;
    }
    if (!imageDimsSet.contains(outputIndex)) {
      expectedOutputShape[outputIndex] = inputShape[i];
    } else {
      expectedOutputShape[outputIndex] = outputTileSize * inputShape[i];
    }
  }
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  return success();
}

SmallVector<Range>
WinogradOutputTransformOp::getIterationDomain(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  SmallVector<Range> loopBounds(getIterationDomainRank());
  int count = 0;
  for (auto dim :
       llvm::seq<int64_t>(imageDimensions().size(), getInputRank())) {
    loopBounds[count].offset = zero;
    loopBounds[count].size = getDimValue(builder, loc, source, dim);
    loopBounds[count].stride = one;
    count++;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType>
WinogradOutputTransformOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult> WinogradOutputTransformOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);
  const int cDim = channelDim();

  assert(offsets.size() == 4);
  const auto hDim = getImageDimensions()[0];
  const auto wDim = getImageDimensions()[1];
  SmallVector<OpFoldResult> inputOffsets(getInputRank(), zero);
  SmallVector<OpFoldResult> outputOffsets(getOutputRank(), zero);

  inputOffsets[2] = outputOffsets[0] = offsets[0];
  inputOffsets[3] = offsets[1];
  inputOffsets[4] = offsets[2];
  inputOffsets[5] = outputOffsets[cDim] = offsets[3];

  SmallVector<OpFoldResult> inputStrides(getInputRank(), one);
  SmallVector<OpFoldResult> outputStrides(getOutputRank(), one);

  ReifiedRankedShapedTypeDims reifiedResultShapes, reifiedInputShapes;
  if (failed(reifyResultShapes(builder, reifiedResultShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> outputSizes = reifiedResultShapes[0];
  if (failed(getStaticOrReifiedInputDims(builder, loc, input(),
                                         reifiedInputShapes))) {
    return failure();
  }
  SmallVector<OpFoldResult> inputSizes = reifiedInputShapes[0];

  inputSizes[2] = outputSizes[0] = sizes[0];
  inputSizes[5] = outputSizes[cDim] = sizes[3];

  assert(sizes.size() == 4);
  inputSizes[2] = outputSizes[0] = sizes[0];
  inputSizes[3] = sizes[1];
  inputSizes[4] = sizes[2];
  inputSizes[5] = outputSizes[cDim] = sizes[3];

  auto hSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizes[1], offsets[1], outputSizes[hDim],
      getOutputTileSize(), getOutputTileSize());
  auto wSizeAndOffset = getScaledSizeAndOffset(
      builder, loc, sizes[2], offsets[2], outputSizes[wDim],
      getOutputTileSize(), getOutputTileSize());

  outputSizes[hDim] = hSizeAndOffset.first;
  outputSizes[wDim] = wSizeAndOffset.first;
  outputOffsets[hDim] = hSizeAndOffset.second;
  outputOffsets[wDim] = wSizeAndOffset.second;

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, input(), inputOffsets, inputSizes, inputStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, output(), outputOffsets,
                                      outputSizes, outputStrides));

  SmallVector<Type, 4> resultTypes;
  if (hasPureTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult WinogradOutputTransformOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    auto resultShape = cast<ShapedType>(output().getType()).getShape();
    resultSizes = getAsOpFoldResult(builder.getIndexArrayAttr(resultShape));
    resultOffsets =
        SmallVector<OpFoldResult>(getOutputRank(), builder.getIndexAttr(0));
    const int cDim = channelDim();
    const auto hDim = getImageDimensions()[0];
    const auto wDim = getImageDimensions()[1];
    auto loc = getLoc();
    resultOffsets[0] = offsets[0];
    resultOffsets[cDim] = offsets[3];
    resultSizes[0] = sizes[0];
    resultSizes[cDim] = sizes[3];
    SmallVector<SmallVector<OpFoldResult>> reifiedResultShapes;
    if (failed(reifyResultShapes(builder, reifiedResultShapes))) {
      return failure();
    }
    auto hSizeAndOffset = getScaledSizeAndOffset(
        builder, loc, sizes[1], offsets[1], reifiedResultShapes[0][hDim],
        getOutputTileSize(), getOutputTileSize());
    auto wSizeAndOffset = getScaledSizeAndOffset(
        builder, loc, sizes[2], offsets[2], reifiedResultShapes[0][wDim],
        getOutputTileSize(), getOutputTileSize());

    resultSizes[hDim] = hSizeAndOffset.first;
    resultSizes[wDim] = wSizeAndOffset.first;
    resultOffsets[hDim] = hSizeAndOffset.second;
    resultOffsets[wDim] = wSizeAndOffset.second;
    return success();
  }
  return failure();
}

LogicalResult WinogradOutputTransformOp::fold(FoldAdaptor,
                                              SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult WinogradOutputTransformOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

/// Utility function to check whether a given ShapedType has the expected rank.
static LogicalResult checkShapeRank(Operation *op, StringRef operandName,
                                    ShapedType shapedType,
                                    int64_t rankToCompareWith) {
  int64_t opRank = shapedType.getRank();
  if (opRank != rankToCompareWith) {
    return op->emitOpError("expected ")
           << operandName << " to have rank " << rankToCompareWith
           << " but found " << opRank;
  }
  return success();
}

LogicalResult AttentionOp::verify() {
  Operation *op = getOperation();

  int numInputs = getNumDpsInputs();
  int numOutputs = getNumDpsInits();

  if (numInputs != 4) {
    return op->emitOpError(
        "expected 4 input operands: Query, Key, Value and Scale");
  }

  if (numOutputs != 1 && numOutputs != 3) {
    return op->emitOpError(
        "expected 1 or 3 output operands: Output, [Max and Sum]");
  }

  bool isTiled = numOutputs == 3;

  int64_t rankToCompareWith;
  if (isTiled) {
    rankToCompareWith = 2;
  } else {
    rankToCompareWith = 3;
  }

  if (!llvm::all_of(llvm::drop_end(getDpsInputs()), [](Value input) {
        return isa<ShapedType>(input.getType());
      })) {
    return op->emitOpError(
        "expected Query, Key, Value inputs to be of shaped type");
  }

  ShapedType queryType = getQueryType();
  ShapedType keyType = getKeyType();
  ShapedType valueType = getValueType();
  ShapedType outputType = getOutputType();
  Type queryElementType = queryType.getElementType();
  Type keyElementType = keyType.getElementType();
  Type valueElementType = valueType.getElementType();
  Type outputElementType = outputType.getElementType();

  FloatType scaleElementType = dyn_cast<FloatType>(getScale().getType());
  if (!scaleElementType) {
    return op->emitOpError("expected scale to be of floating point type");
  }

  if (failed(checkShapeRank(op, "query", queryType, rankToCompareWith))) {
    return failure();
  }
  if (failed(checkShapeRank(op, "key", keyType, rankToCompareWith))) {
    return failure();
  }
  if (failed(checkShapeRank(op, "value", valueType, rankToCompareWith))) {
    return failure();
  }
  if (failed(checkShapeRank(op, "output", outputType, rankToCompareWith))) {
    return failure();
  }
  ArrayRef<int64_t> queryShape = queryType.getShape();
  ArrayRef<int64_t> keyShape = keyType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  SmallVector<int64_t> valueShape(valueType.getShape());
  bool transposeV = getTransposeV();
  if (transposeV) {
    size_t lastIdx = valueShape.size() - 1;
    std::swap(valueShape[lastIdx - 1], valueShape[lastIdx]);
  }
  if (failed(verifyCompatibleShape(keyShape, valueShape))) {
    return op->emitOpError("incompatible value shape");
  }
  if (failed(verifyCompatibleShape(queryShape, outputShape))) {
    return op->emitOpError("incompatible output shape");
  }
  if (queryElementType != keyElementType ||
      queryElementType != valueElementType ||
      queryElementType != scaleElementType) {
    return op->emitOpError(
        "element types of (Q)uery, (K)ey and (V)alue and scale should be "
        "same");
  }
  if (!isTiled) {
    // Vanilla attention.
    if (queryElementType != outputElementType) {
      return op->emitOpError("expected element type for Output ")
             << queryElementType << "but found " << outputElementType
             << " instead";
    }
    if (keyShape[2] != queryShape[2]) {
      return op->emitOpError("query and key head dimension mismatch");
    }
  }
  if (isTiled) {
    // Tiled/Flash attention.
    ShapedType maxType = *getMaxType();
    ShapedType sumType = *getSumType();
    if (failed(checkShapeRank(op, "max", maxType, 1))) {
      return failure();
    }
    if (failed(checkShapeRank(op, "sum", sumType, 1))) {
      return failure();
    }
    Type maxElementType = maxType.getElementType();
    Type sumElementType = sumType.getElementType();
    ArrayRef<int64_t> maxShape = maxType.getShape();
    ArrayRef<int64_t> sumShape = sumType.getShape();
    if (outputElementType != maxElementType ||
        maxElementType != sumElementType) {
      return op->emitOpError(
          "element types of tiled output, max and sum should be same");
    }
    if (failed(verifyCompatibleShape(maxShape, sumShape))) {
      return op->emitOpError("incompatible sum shape");
    }
    if (maxShape[0] != queryShape[0]) {
      return op->emitOpError("Query and max dimension-0 mismatch");
    }
  }

  return success();
}

SmallVector<Range> AttentionOp::getIterationDomain(OpBuilder &builder) {
  int64_t iterationDomainRank = getIterationDomainRank();
  SmallVector<Range> loopBounds(iterationDomainRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getQuery();
  for (auto dim : llvm::seq<int64_t>(0, iterationDomainRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> AttentionOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getIterationDomainRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

FailureOr<TilingResult>
AttentionOp::getTiledImplementation(OpBuilder &builder,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  assert(offsets.size() == getIterationDomainRank());
  assert(sizes.size() == getIterationDomainRank());

  Location loc = getLoc();
  auto one = builder.getIndexAttr(1);
  auto zero = builder.getIndexAttr(0);

  SmallVector<OpFoldResult> queryOutputOffsets(getQueryRank(), zero);
  SmallVector<OpFoldResult> queryOutputStrides(getQueryRank(), one);
  ArrayRef<int64_t> queryShape = getQueryType().getShape();
  SmallVector<OpFoldResult> queryOutputSizes =
      getAsOpFoldResult(builder.getIndexArrayAttr(queryShape));
  for (auto [idx, info] : llvm::enumerate(llvm::zip_equal(offsets, sizes))) {
    queryOutputOffsets[idx] = std::get<0>(info);
    queryOutputSizes[idx] = std::get<1>(info);
  }

  SmallVector<OpFoldResult> keyValueOffsets(getKeyRank(), zero);
  SmallVector<OpFoldResult> keyValueStrides(getKeyRank(), one);
  ArrayRef<int64_t> keyShape = getKeyType().getShape();
  SmallVector<OpFoldResult> keyValueSizes =
      getAsOpFoldResult(builder.getIndexArrayAttr(keyShape));
  keyValueSizes[0] = sizes[0];
  keyValueOffsets[0] = offsets[0];

  Value scale = getScale();

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(getSlice(builder, loc, getQuery(),
                                      queryOutputOffsets, queryOutputSizes,
                                      queryOutputStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, getKey(), keyValueOffsets,
                                      keyValueSizes, keyValueStrides));
  tiledOperands.emplace_back(getSlice(builder, loc, getValue(), keyValueOffsets,
                                      keyValueSizes, keyValueStrides));
  tiledOperands.emplace_back(scale);
  tiledOperands.emplace_back(getSlice(builder, loc, getOutput(),
                                      queryOutputOffsets, queryOutputSizes,
                                      queryOutputStrides));

  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics())
    resultTypes.push_back(tiledOperands[4].getType());

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult AttentionOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    ArrayRef<int64_t> resultShape = getOutputType().getShape();
    resultSizes = getAsOpFoldResult(builder.getIndexArrayAttr(resultShape));
    resultOffsets =
        SmallVector<OpFoldResult>(getOutputRank(), builder.getIndexAttr(0));
    for (auto [idx, info] : llvm::enumerate(llvm::zip_equal(offsets, sizes))) {
      resultOffsets[idx] = std::get<0>(info);
      resultSizes[idx] = std::get<1>(info);
    }
    return success();
  }
  return failure();
}

LogicalResult AttentionOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}

LogicalResult AttentionOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return cast<LinalgExtOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    getEffectsImpl(effects, getDpsInputs(), getDpsInits());                    \
  }

DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(SortOp)
DEFINE_OP_GET_EFFECTS(FftOp)
DEFINE_OP_GET_EFFECTS(ReverseOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(TopkOp)
DEFINE_OP_GET_EFFECTS(PackOp)
DEFINE_OP_GET_EFFECTS(UnPackOp)
DEFINE_OP_GET_EFFECTS(WinogradInputTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradFilterTransformOp)
DEFINE_OP_GET_EFFECTS(WinogradOutputTransformOp)
DEFINE_OP_GET_EFFECTS(AttentionOp)

//===----------------------------------------------------------------------===//
// iree_linalg_ext.set_encoding
//===----------------------------------------------------------------------===//

LogicalResult SetEncodingOp::verify() {
  // Source and the result have the same rank.
  if (getSourceType().getEncoding()) {
    return emitOpError(
        "source of set_encoding op cannot have a tensor encoding");
  }
  if (!getResultType().getEncoding().isa_and_nonnull<EncodingAttr>()) {
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
  reifiedReturnShapes[0] = getDims(builder, getLoc(), getSource());
  return success();
}

//===----------------------------------------------------------------------===//
// iree_linalg_ext.unset_encoding
//===----------------------------------------------------------------------===//

LogicalResult UnsetEncodingOp::verify() {
  if (getResultType().getEncoding()) {
    return emitOpError(
        "result of unset_encoding op cannot have a tensor encoding");
  }
  if (!getSourceType().getEncoding().isa_and_nonnull<EncodingAttr>()) {
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
  return success();
}

LogicalResult UnsetEncodingOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0] = getDims(builder, getLoc(), getSource());
  return success();
}

//===----------------------------------------------------------------------===//
// iree_linalg_ext.encoding
//===----------------------------------------------------------------------===//

EncodingAttr EncodingAttr::get(MLIRContext *ctx, EncodingRole role,
                               ArrayRef<Type> elemTypes, Type origType,
                               std::optional<int64_t> matmulNarrowM,
                               std::optional<int64_t> matmulNarrowN,
                               ArrayRef<AffineMap> maps,
                               ArrayRef<int64_t> roundDimsTo) {
  Builder b(ctx);
  auto optionalToAttr = [&](std::optional<int64_t> x) {
    return x ? b.getIndexAttr(*x) : IntegerAttr();
  };
  auto roleAttr = EncodingRoleAttr::get(ctx, role);
  auto origTypeAttr = origType ? TypeAttr::get(origType) : TypeAttr();
  auto roundDimsToAttr = roundDimsTo.empty()
                             ? DenseI64ArrayAttr()
                             : b.getDenseI64ArrayAttr(roundDimsTo);
  return get(ctx, roleAttr, b.getTypeArrayAttr(elemTypes), origTypeAttr,
             optionalToAttr(matmulNarrowM), optionalToAttr(matmulNarrowN),
             b.getAffineMapArrayAttr(maps), roundDimsToAttr);
}

AffineMap EncodingAttr::getMapForRole() {
  EncodingRole role = getRole().getValue();
  switch (role) {
  case EncodingRole::LHS:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[0]).getAffineMap();
  case EncodingRole::RHS:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[1]).getAffineMap();
  case EncodingRole::RESULT:
    return llvm::cast<AffineMapAttr>(getUserIndexingMaps()[2]).getAffineMap();
  default:
    return AffineMap();
  }
}

unsigned EncodingAttr::mapDimToRoleIndex(int64_t dimPos) {
  AffineMap map = getMapForRole();
  auto idx = map.getResultPosition(getAffineDimExpr(dimPos, getContext()));
  assert(idx.has_value());
  return idx.value();
}

ArrayRef<int64_t> EncodingAttr::getRoundDimsToArray() {
  auto roundDimsTo = getRoundDimsTo();
  if (!roundDimsTo) {
    return {};
  }
  return roundDimsTo.cast<DenseI64ArrayAttr>().asArrayRef();
}

//===---------------------------------------------------------------------===//
// LinalgExt Dialect Helpers
//===---------------------------------------------------------------------===//

EncodingAttr getEncodingAttr(RankedTensorType type) {
  return dyn_cast_or_null<EncodingAttr>(type.getEncoding());
}

FailureOr<linalg::ContractionDimensions>
getEncodingContractionDims(EncodingAttr encoding) {
  auto indexingMapsAttr = encoding.getUserIndexingMaps();
  SmallVector<AffineMap> indexingMaps = llvm::map_to_vector(
      indexingMapsAttr.getValue(), [](Attribute m) -> AffineMap {
        return cast<AffineMapAttr>(m).getAffineMap();
      });
  return linalg::inferContractionDims(indexingMaps);
}

} // namespace mlir::iree_compiler::IREE::LinalgExt

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
