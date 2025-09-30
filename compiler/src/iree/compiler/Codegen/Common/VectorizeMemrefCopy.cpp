// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vectorize-memref-copy"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORIZEMEMREFCOPYPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Moves the copy into a single threaded forall.
static void distributeCopyToSingleThread(RewriterBase &rewriter,
                                         memref::CopyOp copy) {
  scf::ForallOp newForallOp = scf::ForallOp::create(
      rewriter, copy.getLoc(), ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
      /*outputs=*/ValueRange(), /*mapping=*/rewriter.getArrayAttr({}));
  rewriter.moveOpBefore(copy, newForallOp.getBody(),
                        newForallOp.getBody()->begin());
}

// For optimal performance we always want to copy 128 bits
static constexpr int kPreferredCopyNumBits = 128;

static SmallVector<OpFoldResult> getCopyTileSizes(Builder &b,
                                                  memref::CopyOp copy) {
  int64_t rank = copy.getTarget().getType().getRank();
  if (rank == 0) {
    return {};
  }

  SmallVector<OpFoldResult> tileSizes(rank - 1, b.getIndexAttr(1));
  int64_t elementBitWidth = llvm::cast<MemRefType>(copy.getTarget().getType())
                                .getElementTypeBitWidth();
  tileSizes.push_back(b.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));
  return tileSizes;
}

/// Distributes a copy with a thread mapping.
static void copyDynamicDimsToLoops(RewriterBase &rewriter, memref::CopyOp copyOp,
                                   ArrayRef<OpFoldResult> tileSizes) {   
  int64_t rank = tileSizes.size();
  assert(rank == copyOp.getTarget().getType().getRank() &&
         "tile size and copy rank mismatch");
  // if (rank == 0) {
  //   distributeCopyToSingleThread(rewriter, copy);
  //   return;
  // }

  Location loc = copyOp.getLoc();
  // MLIRContext *context = rewriter.getContext();

  SmallVector<OpFoldResult> bounds =
      memref::getMixedSizes(rewriter, loc, copyOp.getSource());

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  MemRefType type = cast<MemRefType>(copyOp.getTarget().getType());

  for (auto [size, b] : llvm::zip_equal(tileSizes, bounds)) {
    LLVM_DEBUG(llvm::dbgs() << "size: " << size << "\n");
    LLVM_DEBUG(llvm::dbgs() << "bound: " << b << "\n");
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, b);
    Value step = getValueOrCreateConstantIndexOp(rewriter, loc, size);
    LLVM_DEBUG(llvm::dbgs() << "TEST\n");
    auto newForOp = scf::ForOp::create(
        rewriter, loc, lb, ub, step, ValueRange{});
    LLVM_DEBUG(llvm::dbgs() << "newForOp: " << newForOp << "\n");
    offsets.push_back(newForOp.getInductionVar());
    sizes.push_back(size);
    strides.push_back(rewriter.getIndexAttr(1));
    rewriter.setInsertionPointToStart(newForOp.getBody());
    // if (ShapedType::isDynamic(size)) {
    //   Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    //   Value ub = dyn_cast<Value>(b);
    //   Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    //   auto newForOp = scf::ForOp::create(
    //       rewriter, loc, lb, ub, step, ValueRange{});
    //   offsets.push_back(newForOp.getInductionVar());
    //   sizes.push_back(rewriter.getIndexAttr(1));
    //   rewriter.setInsertionPointToStart(newForOp.getBody());
    // } else {
    //   offsets.push_back(rewriter.getIndexAttr(0));
    //   sizes.push_back(rewriter.getIndexAttr(size));
    // }
    // strides.push_back(rewriter.getIndexAttr(1));
  }
  Value sourceTile = memref::SubViewOp::create(rewriter, loc, copyOp.getSource(),
                                               offsets, sizes, strides);
  Value targetTile = memref::SubViewOp::create(rewriter, loc, copyOp.getTarget(),
                                               offsets, sizes, strides);
  rewriter.replaceOpWithNewOp<memref::CopyOp>(copyOp, sourceTile, targetTile);
}

struct ConvertLinalgCopyToMemrefCopy final : OpRewritePattern<linalg::CopyOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.hasPureTensorSemantics()) {
      return failure();
    }
    memref::CopyOp::create(rewriter, copyOp.getLoc(),
                           copyOp.getDpsInputOperand(0)->get(),
                           copyOp.getDpsInitOperand(0)->get());
    rewriter.eraseOp(copyOp);
    return success();
  }
};

struct VectorizeMemrefCopyPass final
    : impl::VectorizeMemrefCopyPassBase<VectorizeMemrefCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    SmallVector<memref::CopyOp> copies;

    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within thread/warp/lane foralls
    // to be skipped.
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      // if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
      //   // Skip ops not within forall ops with thread/warp/lane mappings.
      //   if (!forallOpHasMappingType<IREE::GPU::LaneIdAttr,
      //                              gpu::GPUWarpMappingAttr,
      //                              gpu::GPUThreadMappingAttr>(forallOp)) {
      //     return WalkResult::skip();
      //   }
      // }
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        copies.push_back(copy);
      }
      return WalkResult::advance();
    });

    IRRewriter rewriter(ctx);
    for (auto copy : copies) {
      LLVM_DEBUG(llvm::dbgs() << "copy: " << copy << "\n");
      rewriter.setInsertionPoint(copy);
      SmallVector<OpFoldResult> tileSizes = getCopyTileSizes(rewriter, copy);
      copyDynamicDimsToLoops(rewriter, copy, tileSizes);
    }

    RewritePatternSet patterns(ctx);
    patterns.add<linalg::CopyVectorizationPattern>(&getContext());
    patterns.add<ConvertLinalgCopyToMemrefCopy>(&getContext());
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler
