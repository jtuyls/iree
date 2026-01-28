// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MATERIALIZEENCODINGSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

/// Information extracted from a TensorEncodeOp for creating the executable.
struct EncodingInfo {
  Location loc;
  Type sourceEncoding;
  Type resultEncoding;
  // These are copied, not references.
  SmallVector<Value> sourceEncodingDims;
  SmallVector<Value> resultEncodingDims;

  EncodingInfo(Location loc) : loc(loc) {}
};

struct MaterializeEncodingsPass
    : public IREE::Stream::impl::MaterializeEncodingsPassBase<
          MaterializeEncodingsPass> {
  void runOnOperation() override;
};
} // namespace

/// Returns a pretty function name based on the encoding types.
static std::string getDispatchFuncName(const EncodingInfo &info) {
  std::string str;
  llvm::raw_string_ostream os(str);
  auto printShape = [&](Type type) {
    auto rankedType = dyn_cast<RankedTensorType>(type);
    if (!rankedType)
      return;
    for (auto dimSize : rankedType.getShape()) {
      if (ShapedType::isDynamic(dimSize)) {
        os << "D";
      } else {
        os << std::to_string(dimSize);
      }
      os << "x";
    }
    rankedType.getElementType().print(os);
  };

  os << "encode_";
  printShape(info.sourceEncoding);
  os << "_to_";
  printShape(info.resultEncoding);
  return str;
}

/// Creates a workgroup function for the encoding operation.
static func::FuncOp createWorkgroupFunc(const EncodingInfo &info,
                                        StringRef functionName) {
  Location loc = info.loc;
  MLIRContext *ctx = loc.getContext();
  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocs;
  auto bindingType = IREE::Stream::BindingType::get(ctx);
  int ordinalCount = 0;

  // Add the block argument for the source resource and corresponding dynamic
  // dimension sizes.
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);
  for (size_t i = 0; i < info.sourceEncodingDims.size(); ++i) {
    argumentTypes.push_back(IndexType::get(ctx));
    argumentLocs.push_back(loc);
  }

  // Add the block argument for the result resource and corresponding dynamic
  // dimension sizes.
  for (size_t i = 0; i < info.resultEncodingDims.size(); ++i) {
    argumentTypes.push_back(IndexType::get(ctx));
    argumentLocs.push_back(loc);
  }
  argumentTypes.push_back(bindingType);
  argumentLocs.push_back(loc);

  // Build function type matching the region signature.
  auto functionType = FunctionType::get(ctx, argumentTypes, /*results=*/{});
  auto funcOp = mlir::func::FuncOp::create(loc, functionName, functionType);
  Block &block = funcOp.getBody().emplaceBlock();
  block.addArguments(argumentTypes, argumentLocs);
  OpBuilder builder(funcOp.getBody());

  // Build operations to handle load/store from/to the bindings.
  SmallVector<Value> sourceDynamicDims;
  SmallVector<Value> destinationDynamicDims;
  for (auto argument : block.getArguments().drop_front(1).take_front(
           info.sourceEncodingDims.size())) {
    sourceDynamicDims.push_back(
        IREE::TensorExt::DispatchWorkloadOrdinalOp::create(
            builder, loc, argument, builder.getIndexAttr(ordinalCount++)));
  }
  for (auto argument : block.getArguments().drop_back(1).take_back(
           info.resultEncodingDims.size())) {
    destinationDynamicDims.push_back(
        IREE::TensorExt::DispatchWorkloadOrdinalOp::create(
            builder, loc, argument, builder.getIndexAttr(ordinalCount++)));
  }

  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
  auto sourceDispatchType = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, info.sourceEncoding);
  Value source = IREE::Stream::BindingSubspanOp::create(
      builder, loc, sourceDispatchType, block.getArgument(0), zero,
      sourceDynamicDims);
  auto destinationDispatchType = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, info.resultEncoding);
  Value destination = IREE::Stream::BindingSubspanOp::create(
      builder, loc, destinationDispatchType, block.getArguments().back(), zero,
      destinationDynamicDims);

  // Load the value from the source binding.
  RankedTensorType sourceType = sourceDispatchType.asRankedTensorType();
  Value value = IREE::TensorExt::DispatchTensorLoadOp::create(
      builder, loc, sourceType, source, sourceDynamicDims);

  // We can only add/remove encodings using set_encoding/unset_encoding ops
  // today. Thus, we firstly need to bring the tensor encodings to pure tensor
  // types, and then encode the tensor types when needed.
  RankedTensorType destinationType =
      destinationDispatchType.asRankedTensorType();
  if (sourceType != destinationType) {
    if (sourceType.getEncoding()) {
      value = IREE::Encoding::UnsetEncodingOp::create(
          builder, loc, sourceType.dropEncoding(), value, sourceDynamicDims,
          /*encoding_dims=*/ValueRange{});
    }
    if (destinationType.getEncoding()) {
      value = IREE::Encoding::SetEncodingOp::create(
          builder, loc, destinationType, value, /*encoding_dims=*/ValueRange{});
    }
  }

  // Store the value to the destination binding.
  IREE::TensorExt::DispatchTensorStoreOp::create(
      builder, loc, value, destination, destinationDynamicDims);
  func::ReturnOp::create(builder, loc);

  return funcOp;
}

/// Creates an export op pointing at the `funcOp` function.
static IREE::Stream::ExecutableExportOp
createExportOp(RewriterBase &rewriter, Location loc, const EncodingInfo &info,
               IREE::Stream::ExecutableOp executableOp, func::FuncOp funcOp) {
  SmallVector<Type> workloadTypes;
  SmallVector<Location> workloadLocs;
  for (size_t i = 0; i < info.sourceEncodingDims.size(); ++i) {
    workloadTypes.push_back(IndexType::get(loc.getContext()));
    workloadLocs.push_back(loc);
  }
  for (size_t i = 0; i < info.resultEncodingDims.size(); ++i) {
    workloadTypes.push_back(IndexType::get(loc.getContext()));
    workloadLocs.push_back(loc);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&executableOp.getBody().front());
  auto exportOp = IREE::Stream::ExecutableExportOp::create(
      rewriter, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *block = rewriter.createBlock(&exportOp.getWorkgroupCount(),
                                      exportOp.getWorkgroupCount().end(),
                                      workloadTypes, workloadLocs);
  rewriter.setInsertionPointToStart(block);
  auto defaultCountOp =
      IREE::TensorExt::DispatchWorkgroupCountFromSliceOp::create(
          rewriter, loc, block->getArguments());
  IREE::Stream::ReturnOp::create(rewriter, loc, defaultCountOp.getResults());
  return exportOp;
}

/// Creates the executable op and build the content for the encoding operation.
static std::pair<IREE::Stream::ExecutableOp, IREE::Stream::ExecutableExportOp>
createExecutableAndExport(RewriterBase &rewriter, ModuleOp moduleOp,
                          const EncodingInfo &info, int executableId) {
  OpBuilder::InsertionGuard guard(rewriter);

  rewriter.setInsertionPoint(&moduleOp.getBody()->back());
  Location loc = info.loc;
  std::string executableName = "_encoding_" + std::to_string(executableId);
  auto executableOp =
      IREE::Stream::ExecutableOp::create(rewriter, loc, executableName);
  // Move to beginning of module, before any initializers or functions.
  executableOp.getOperation()->moveBefore(&moduleOp.getBody()->front());
  executableOp.setPrivate();

  // Build the inner module and func op, and an export op pointing at the
  // function.
  std::string funcName = executableName + "_" + getDispatchFuncName(info);
  auto funcOp = createWorkgroupFunc(info, funcName);
  rewriter.setInsertionPointToStart(&executableOp.getBody().front());
  auto innerModule = mlir::ModuleOp::create(rewriter, loc);
  innerModule.push_back(funcOp);
  IREE::Stream::ExecutableExportOp exportOp =
      createExportOp(rewriter, loc, info, executableOp, funcOp);
  return std::make_pair(executableOp, exportOp);
}

/// Returns the encoding signature for dispatch as ArrayAttr form.
static ArrayAttr getEncodingSignature(Builder &builder,
                                      const EncodingInfo &info) {
  return builder.getArrayAttr(
      {TypeAttr::get(info.sourceEncoding), TypeAttr::get(info.resultEncoding)});
}

/// Replaces the encoding dispatch region with an `AsyncDispatchOp`.
static void replaceEncodingDispatchWithAsyncDispatch(
    IRRewriter &rewriter, IREE::Stream::HoistableDispatchOp dispatchOp,
    const EncodingInfo &info, IREE::Stream::ExecutableOp executableOp,
    IREE::Stream::ExecutableExportOp exportOp) {
  rewriter.setInsertionPoint(dispatchOp);
  Location loc = dispatchOp.getLoc();

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> operandOffsets = {zero};
  SmallVector<Value> operandEnds = {dispatchOp.getInputSizes()[0]};
  SmallVector<Value> operandLengths = {dispatchOp.getInputSizes()[0]};
  SmallVector<Value> operands = {dispatchOp.getInputs()[0]};
  for (auto argument : info.sourceEncodingDims) {
    operands.push_back(argument);
  }
  for (auto argument : info.resultEncodingDims) {
    operands.push_back(argument);
  }

  SmallVector<int64_t> tiedArguments = {
      IREE::Util::TiedOpInterface::kUntiedIndex};
  SmallVector<Value> dynamicDims;
  for (Value argument : info.sourceEncodingDims) {
    dynamicDims.push_back(argument);
  }
  for (Value argument : info.resultEncodingDims) {
    dynamicDims.push_back(argument);
  }

  rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
      dispatchOp, exportOp,
      /*workload=*/dynamicDims, dispatchOp.getResult(0).getType(), operands,
      dispatchOp.getInputSizes()[0], operandOffsets, operandEnds,
      operandLengths, dispatchOp.getResultSizes()[0], tiedArguments,
      dispatchOp.getAffinityAttr());
}

/// Extracts encoding info from a TensorEncodeOp inside an
/// HoistableDispatchOp.
static std::optional<EncodingInfo>
extractEncodingInfo(IREE::Stream::HoistableDispatchOp dispatchOp) {
  IREE::Stream::TensorEncodeOp encodeOp;
  dispatchOp.getBody().walk([&](IREE::Stream::TensorEncodeOp op) {
    encodeOp = op;
  });
  if (!encodeOp) {
    return std::nullopt;
  }

  EncodingInfo info(dispatchOp.getLoc());
  info.sourceEncoding = encodeOp.getSourceEncoding();
  info.resultEncoding = encodeOp.getResultEncoding();
  // Copy dim values - they should be values defined outside the dispatch
  // region.
  for (Value dim : encodeOp.getSourceEncodingDims()) {
    info.sourceEncodingDims.push_back(dim);
  }
  for (Value dim : encodeOp.getResultEncodingDims()) {
    info.resultEncodingDims.push_back(dim);
  }
  return info;
}

void MaterializeEncodingsPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ModuleOp moduleOp = getOperation();

  // Collect all encoding dispatch region ops by walking the entire module.
  SmallVector<IREE::Stream::HoistableDispatchOp> dispatchOps;
  moduleOp.walk([&](IREE::Stream::HoistableDispatchOp op) {
    dispatchOps.push_back(op);
  });

  if (dispatchOps.empty()) {
    return;
  }

  // Mapping from (sourceEncoding, resultEncoding) to the executable op and the
  // export op. The encoding changes are described by the encoding pairs and the
  // executables can be reused.
  DenseMap<ArrayAttr, std::pair<IREE::Stream::ExecutableOp,
                                IREE::Stream::ExecutableExportOp>>
      cachedExecutables;

  IRRewriter rewriter(ctx);
  int executableId = 0;

  // Process each dispatch op. We collected the ops first so we can safely
  // modify them while iterating.
  for (auto dispatchOp : dispatchOps) {
    // Extract encoding info.
    auto maybeInfo = extractEncodingInfo(dispatchOp);
    if (!maybeInfo) {
      dispatchOp.emitError("expected TensorEncodeOp inside encoding dispatch region");
      return signalPassFailure();
    }
    EncodingInfo info = std::move(*maybeInfo);

    ArrayAttr encodingSignature = getEncodingSignature(rewriter, info);
    if (!cachedExecutables.contains(encodingSignature)) {
      cachedExecutables[encodingSignature] =
          createExecutableAndExport(rewriter, moduleOp, info, executableId++);
    }
    auto [executableOp, exportOp] = cachedExecutables[encodingSignature];
    replaceEncodingDispatchWithAsyncDispatch(rewriter, dispatchOp, info,
                                             executableOp, exportOp);
  }
}

} // namespace mlir::iree_compiler::IREE::Stream
