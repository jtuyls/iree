// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-distribute"

using LayoutDimension = mlir::iree_compiler::IREE::VectorExt::LayoutDimension;
using LayoutDimensionAttr =
    mlir::iree_compiler::IREE::VectorExt::LayoutDimensionAttr;
using VectorLayoutInterface =
    mlir::iree_compiler::IREE::VectorExt::VectorLayoutInterface;
using PerDimLayoutAttr = mlir::iree_compiler::IREE::VectorExt::PerDimLayoutAttr;
using LayoutAttr = mlir::iree_compiler::IREE::VectorExt::LayoutAttr;

namespace mlir::iree_compiler {

namespace {

// Vector layout option setter aimed at contractions. Currently this only sets
// anchors for two types of operations; vector.contract and vector.transfer_read
// from non-shared memory. The assumption in this case is that all IR input to
// this pass has a leaf rooted on a transfer_read or includes a contraction in
// the program slice, meaning all operations should receive layouts. Layout
// setting for other problems like reductions is TODO.
class ContractionVectorLayoutOptions : public VectorLayoutOptions {
public:
  ContractionVectorLayoutOptions(Operation *root,
                                 ArrayRef<int64_t> workgroupSize,
                                 IREE::GPU::MMAScheduleAttr schedule,
                                 Value laneId, int64_t subgroupSize,
                                 bool printLayout)
      : VectorLayoutOptions(root, /*fullConversion=*/!printLayout),
        workgroupSize(workgroupSize), schedule(schedule),
        printLayout(printLayout), patterns(root->getContext()) {
    populateGPUDistributionPatterns(patterns);
    populateGPUDistributionLayoutAttrPatterns(laneId, patterns);
    populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId,
                                                  subgroupSize);
  }

  LogicalResult setAnchorOps(VectorLayoutAnalysis &analysis) override {
    MLIRContext *context = root->getContext();
    WalkResult walkResult = root->walk([&](Operation *op) {
      LogicalResult setResult =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case([&](vector::ContractionOp contract) {
                return setContractionAnchor(context, analysis, contract);
              })
              .Case([&](vector::TransferReadOp transfer) {
                return setTransferReadAnchor(context, analysis, transfer);
              })
              .Default([](Operation *) { return success(); });
      return failed(setResult) ? WalkResult::interrupt()
                               : WalkResult::advance();
    });
    return failure(walkResult.wasInterrupted());
  }

  RewritePatternSet &getPatterns() { return patterns; }

private:
  // Sets an anchoring layout for the given contraction op. Looks for a
  // supported mma type from the cached list of mma types and populates the
  // necessary distribution pattern for those contractions.
  LogicalResult setContractionAnchor(MLIRContext *context,
                                     VectorLayoutAnalysis &analysis,
                                     vector::ContractionOp contract) {
    // TODO: Add SIMT fallback.
    if (!schedule) {
      return contract->emitError("missing mma schedule for contraction");
    }

    auto layouts = schedule.getContractionLayout(contract);
    if (!layouts) {
      return contract->emitError("cannot get concrete layout for contraction");
    }

    auto [aLayout, bLayout, cLayout] = *layouts;
    if (analysis.setAnchor(contract.getLhs(), aLayout).failed()) {
      return failure();
    }
    if (analysis.setAnchor(contract.getRhs(), bLayout).failed()) {
      return failure();
    }
    if (analysis.setAnchor(contract.getAcc(), cLayout).failed()) {
      return failure();
    }
    if (analysis.setAnchor(contract.getResult(), cLayout).failed()) {
      return failure();
    }
    contract->setAttr("iree.amdgpu.mma", schedule.getIntrinsic());
    if (printLayout) {
      llvm::outs() << "contract A vector layout: " << aLayout << "\n";
      llvm::outs() << "contract B vector layout: " << bLayout << "\n";
      llvm::outs() << "contract C vector layout: " << cLayout << "\n";
    }
    LLVM_DEBUG({
      llvm::dbgs() << "chosen a layout: " << aLayout << "\n";
      llvm::dbgs() << "chosen b layout: " << bLayout << "\n";
      llvm::dbgs() << "chosen c layout: " << cLayout << "\n";
      llvm::dbgs() << "anchor set on contract: " << contract << "\n";
    });

    if (isa<IREE::GPU::MMAAttr>(schedule.getIntrinsic())) {
      if (!populatedMma) {
        populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
        populatedMma = true;
      }
    } else {
      llvm_unreachable("Unsupported mma type");
    }
    return success();
  }

  // Sets a layout anchor for reads from global memory.
  // The layout this generates is approximately the following:
  //
  // #layout = #iree_vector_ext.nested_layout<
  //    subgroups_per_workgroup = [1, ..., 1]
  //    batches_per_subgroup =    [<remaining undistributed elements>]
  //    outers_per_batch =        [1, ..., 1]
  //    threads_per_outer =       [<greedy from innermost memref dim>]
  //    elements_per_thread =     [1, ..., 128/element_bitwidth, ..., 1]
  //            innermost_memref_dimension ^^^^^^
  //
  // (All orders are the same)
  //    *_order = [<broadcasted_dims>, <transfer_permutation>]>
  //
  // So for the following transfer_read with 64 threads:
  //  vector.transfer_read ... : memref<16x256xf16>, vector<16x32xf16>
  //
  // We use the following layout:
  // #layout = #iree_vector_ext.nested_layout<
  //    subgroups_per_workgroup = [1, 1]
  //    batches_per_subgroup =    [1, 1]
  //    outers_per_batch =        [1, 1]
  //    threads_per_outer =       [16, 4]
  //    elements_per_thread =     [1, 8]
  //
  //    *_order = [0, 1]>
  LogicalResult setTransferReadAnchor(MLIRContext *context,
                                      VectorLayoutAnalysis &analysis,
                                      vector::TransferReadOp transfer) {

    // Get the forward slice of the transfer to approximate whether it will take
    // the layout of a contraction instead. Transfer_read ops used directly by a
    // contraction (i.e. without a copy to shared memory in between) should take
    // the layout of the contraction op. This is common for cases where the
    // initial values of the accumulator in a linalg.matmul is read from memory
    // instead of just being a zerofill.
    ForwardSliceOptions forwardOptions;
    forwardOptions.filter = [&](Operation *op) -> bool {
      return llvm::any_of(op->getResultTypes(), llvm::IsaPred<VectorType>);
    };
    BackwardSliceOptions backwardOptions;
    backwardOptions.filter = [&](Operation *op) -> bool {
      return llvm::any_of(op->getOperandTypes(), llvm::IsaPred<VectorType>);
    };
    SetVector<Operation *> slice =
        getSlice(transfer, backwardOptions, forwardOptions);

    if (llvm::any_of(slice, llvm::IsaPred<vector::ContractionOp>)) {
      return success();
    }

    // Shared memory loads are expected to take the layout of the contraction.
    auto sourceMemRefType =
        dyn_cast<MemRefType>(transfer.getSource().getType());
    if (!sourceMemRefType || hasSharedMemoryAddressSpace(sourceMemRefType)) {
      return success();
    }

    // Take on layout of broadcast.
    if (transfer->hasOneUse() &&
        dyn_cast<vector::BroadcastOp>(*transfer->getUsers().begin())) {
      return success();
    }

    // TODO: Support masking.
    if (transfer.getMask()) {
      transfer->emitOpError(
          "Anchoring on transfer_read with masks is not yet implemented.");
      return failure();
    }

    int64_t bitWidth = IREE::Util::getTypeBitWidth(
        getElementTypeOrSelf(transfer.getVectorType()));
    if (!llvm::isPowerOf2_64(bitWidth) || bitWidth > 128) {
      transfer->emitOpError(
          "Anchoring on transfer_read with element type of bitwidth " +
          std::to_string(bitWidth) + " is not yet implemented");
      return failure();
    }
    int64_t numElementsPerThread = 128 / bitWidth;
    int64_t flatNumElements =
        ShapedType::getNumElements(transfer.getVectorType().getShape());
    int64_t flatNumThreads = ShapedType::getNumElements(workgroupSize);
    if (flatNumElements % flatNumThreads != 0) {
      transfer->emitOpError(
          "Anchoring on transfer_read with unsupported number of elements (not "
          "divisible by workgroup size)");
      return failure();
    }
    numElementsPerThread =
        std::min(numElementsPerThread, flatNumElements / flatNumThreads);

    AffineMap transferMap = transfer.getPermutationMap();
    if (transferMap.getNumDims() == 0) {
      transfer->emitOpError("Anchoring on transfer_read with zero-rank "
                            "permutation map is not supported.");
      return failure();
    }

    // Select the innermost dim of the memref as the contiguous dim to load
    // from.
    int64_t transferRank = transfer.getVectorType().getRank();
    std::optional<unsigned> maybeDim = transferMap.getResultPosition(
        getAffineDimExpr(transferMap.getNumDims() - 1, context));
    int64_t distXDim = maybeDim ? *maybeDim : transferRank - 1;

    ArrayRef<int64_t> vectorShape = transfer.getVectorType().getShape();

    // Limit the maximum inner vector read width to the innermost contiguous
    // dimension. We could try to be clever and extend this to adjacent
    // dimensions in cases where the innermost read vector dimension is small,
    // but that requires comparing memref strides and is uncommon. For now
    // prioritize warp contiguity over 128-bit read granularity.
    numElementsPerThread =
        std::min(numElementsPerThread, vectorShape[distXDim]);

    llvm::SetVector<unsigned> vectorDimDistributionOrder;
    // Get the order in which to distribute vector dimensions to threads, going
    // from innermost to outermost memref dimension. It's important to note
    // that this heuristic only applies to matrix multiplication cases where
    // we are promoting the operands of a contraction to shared memory and we
    // have no producers fused with the matmul. In general there is no universal
    // way to set an anchoring layout for reads without doing an analysis of how
    // the read values are used.
    for (int i = transferMap.getNumDims() - 1; i >= 0; --i) {
      std::optional<unsigned> maybeDim =
          transferMap.getResultPosition(getAffineDimExpr(i, context));
      if (maybeDim) {
        vectorDimDistributionOrder.insert(*maybeDim);
      }
    }
    // Add all remaining (broadcasted) dimensions
    for (auto dim : llvm::seq(static_cast<int64_t>(0), transferRank)) {
      if (!vectorDimDistributionOrder.contains(dim))
        vectorDimDistributionOrder.insert(dim);
    }

    int64_t residualThreads = flatNumThreads;
    int64_t residualElements = numElementsPerThread;

    SmallVector<int64_t> order(vectorDimDistributionOrder.rbegin(),
                               vectorDimDistributionOrder.rend());

    // Distribute all threads in the workgroup to the "threads" dimension,
    // meaning subgroup counts is unit here, even though the read is being
    // distributed to multiple subgroups. This is in an attempt to do a
    // workgroup contiguous load.
    SmallVector<int64_t> subgroupCounts(transferRank, 1);
    SmallVector<int64_t> batchSizes(transferRank, 1);
    SmallVector<int64_t> outerSizes(transferRank, 1);
    SmallVector<int64_t> threadCounts(transferRank, 1);
    SmallVector<int64_t> elementSizes(transferRank, 1);

    for (auto dim : llvm::reverse(order)) {
      int64_t vectorSize = vectorShape[dim];
      // Set the element count for the innermost vector dimension.
      if (residualElements != 1) {
        elementSizes[dim] = residualElements;
        vectorSize /= residualElements;
        residualElements = 1;
      }

      assert((residualThreads % vectorSize == 0 ||
              vectorSize % residualThreads == 0) &&
             "dividing threads to incompatible vector");
      if (residualThreads <= vectorSize) {
        vectorSize /= residualThreads;
        threadCounts[dim] = residualThreads;
        residualThreads = 1;
      } else {
        residualThreads /= vectorSize;
        threadCounts[dim] = vectorSize;
        vectorSize = 1;
      }

      batchSizes[dim] = vectorSize;
    }

    // Note that the layout setting logic here necessarily uses all threads in
    // the workgroup to perform the read. As a result we can always directly
    // use the counts as the basis for computing the subgroup/thread indices.
    SmallVector<int64_t> subgroupBasis = subgroupCounts;
    SmallVector<int64_t> threadBasis = threadCounts;

    auto layout = IREE::VectorExt::NestedLayoutAttr::get(
        context, subgroupCounts, order, batchSizes, outerSizes, threadCounts,
        order, elementSizes, subgroupBasis,
        SmallVector<bool>(subgroupBasis.size(), true), threadBasis,
        SmallVector<bool>(threadBasis.size(), true));
    if (analysis.setAnchor(transfer.getResult(), layout).failed()) {
      return failure();
    }
    if (printLayout) {
      llvm::outs() << "transfer '" << transfer << "' vector layout: " << layout
                   << "\n";
    }
    return success();
  }

  SmallVector<int64_t, 3> workgroupSize;
  IREE::GPU::MMAScheduleAttr schedule;
  // Whether to print the chosen layout for testing purposes
  bool printLayout;

  bool populatedMma = false;
  RewritePatternSet patterns;
};

struct LLVMGPUVectorDistributePass
    : public LLVMGPUVectorDistributeBase<LLVMGPUVectorDistributePass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<amdgpu::AMDGPUDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    std::array<int64_t, 3> workgroupSize;
    if (func->hasAttr("workgroup_size")) {
      auto tmpSizes =
          llvm::cast<ArrayAttr>(func->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = llvm::cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(func);
      if (!maybeWorkgroupSize) {
        func->emitOpError()
            << "unable to query workgroup_size information from entry point";
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(maybeWorkgroupSize.value())) {
        workgroupSize[index] = value;
      }
      for (auto index : llvm::seq<size_t>(maybeWorkgroupSize->size(), 3)) {
        workgroupSize[index] = 1;
      }
    }

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
      scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
          configDict.get(scheduleAttrName));
    }

    AffineExpr x, y, z;
    bindSymbols(func.getContext(), x, y, z);
    // Construct the expression for linearizing the thread indices.
    AffineExpr linearId =
        x + workgroupSize[0] * y + workgroupSize[1] * workgroupSize[0] * z;

    OpBuilder builder(func);
    builder.setInsertionPointToStart(&func.getFunctionBody().front());
    SmallVector<OpFoldResult> threadGrid = {
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::x),
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(), gpu::Dimension::y),
        builder.createOrFold<gpu::ThreadIdOp>(func.getLoc(),
                                              gpu::Dimension::z)};

    Value linearThreadIdVal = affine::makeComposedAffineApply(
        builder, func.getLoc(), linearId, threadGrid);

    std::optional<int64_t> subgroupSize = getSubgroupSize(func);
    if (!subgroupSize) {
      func->emitOpError()
          << "unable to query subgroup size information from entry point";
      return signalPassFailure();
    }

    ContractionVectorLayoutOptions options(func, workgroupSize, scheduleAttr,
                                           linearThreadIdVal,
                                           subgroupSize.value(), testLayout);
    if (failed(distributeVectorOps(func, options.getPatterns(), options))) {
      func->emitOpError() << "failed to distribute";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUVectorDistribute() {
  return std::make_unique<LLVMGPUVectorDistributePass>();
}

} // namespace mlir::iree_compiler
