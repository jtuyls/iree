// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingPatterns.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-insert-encoding-specialization-markers"

#define GEN_PASS_DEF_INSERTENCODINGSPECIALIZATIONMARKERSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Extracts all unique DynamicLayoutSpecializerAttr from an operation's
/// operands and results.
static SmallVector<IREE::Encoding::DynamicLayoutSpecializerAttr>
getSpecializerAttrs(Operation *op) {
  DenseSet<Attribute> seen;
  SmallVector<IREE::Encoding::DynamicLayoutSpecializerAttr> attrs;

  auto addAttrFromType = [&](Type type) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    if (!tensorType || !tensorType.getEncoding()) {
      return;
    }
    auto attr = dyn_cast<IREE::Encoding::DynamicLayoutSpecializerAttr>(
        tensorType.getEncoding());
    if (attr && attr.supportsSpecialization() && !seen.contains(attr)) {
      attrs.push_back(attr);
      seen.insert(attr);
    }
  };

  // Check operands
  for (Value operand : op->getOperands()) {
    addAttrFromType(operand.getType());
  }

  // Check results
  for (Value result : op->getResults()) {
    addAttrFromType(result.getType());
  }

  return attrs;
}

/// Creates iree_encoding.dim ops for the dimensions that need specialization,
/// wraps them with util.specialize ops.
/// Returns the created specialize ops for post-reification processing.
static void insertMarkersForOp(
    Operation *op, OpBuilder &builder,
    DenseSet<std::pair<Value, unsigned>> &seenKeys,
    SmallVectorImpl<IREE::Util::SpecializeOp> &specializeOps) {
  // Get all specializer attributes from this operation
  auto specializerAttrs = getSpecializerAttrs(op);
  if (specializerAttrs.empty()) {
    return;
  }

  // For each specializer attribute, get the operands to specialize on
  for (auto specializerAttr : specializerAttrs) {
    auto specOperands = specializerAttr.getSpecializationOperands(op);
    if (specOperands.empty()) {
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Creating markers for op: " << *op << "\n");

    // Set insertion point BEFORE the operation - specialize markers should
    // precede the op that uses the specialized value
    builder.setInsertionPoint(op);

    for (const auto &specOp : specOperands) {
      auto key = std::make_pair(specOp.operand, specOp.dimIndex);

      // Check if we've already created markers for this (operand, dimIndex)
      if (seenKeys.contains(key)) {
        LLVM_DEBUG(llvm::dbgs() << "  Skipping duplicate for operand dim "
                                << specOp.dimIndex << "\n");
        continue;
      }
      seenKeys.insert(key);

      // Create the iree_encoding.dim op to query this dimension
      auto encodingDimOp = IREE::Encoding::DimOp::create(
          builder, op->getLoc(), specOp.operand,
          static_cast<int64_t>(specOp.dimIndex));

      LLVM_DEBUG(llvm::dbgs() << "  Created encoding_dim for operand dim "
                              << specOp.dimIndex << ": "
                              << encodingDimOp.getResult() << "\n");

      // Wrap with util.specialize to mark for specialization
      auto specializeOp = IREE::Util::SpecializeOp::create(
          builder, op->getLoc(), encodingDimOp.getResult());
      specializeOps.push_back(specializeOp);

      LLVM_DEBUG(llvm::dbgs() << "  Created util.specialize: "
                              << specializeOp.getResult() << "\n");
    }
  }
}

/// Check if a value is a constant (e.g., from arith.constant).
/// TODO(#...): This is a workaround. The encoding_dims on set_encoding/
/// unset_encoding ops currently include both static and dynamic dimensions.
/// We should update encoding_dims to only contain dynamic dimensions, which
/// would eliminate the need for this constant check. Once that's done,
/// this function and the filtering logic in insertMarkersForSetEncoding
/// can be removed.
static bool isConstantValue(Value value) {
  if (!value.getDefiningOp()) {
    return false;
  }
  return matchPattern(value, m_Constant());
}

/// Creates util.specialize ops for encoding_dims of set_encoding ops.
/// This is a special case: we want to mark the encoding_dims for specialization
/// even before they get consumed by the encoding operation.
/// Only creates markers for dynamic (non-constant) dimensions.
static void insertMarkersForSetEncoding(
    IREE::Encoding::SetEncodingOp setEncodingOp, OpBuilder &builder,
    DenseSet<std::pair<Value, unsigned>> &seenKeys,
    SmallVectorImpl<IREE::Util::SpecializeOp> &specializeOps) {
  // Check if this encoding supports specialization
  auto resultType = setEncodingOp.getResultType();
  auto specializerAttr =
      dyn_cast_or_null<IREE::Encoding::DynamicLayoutSpecializerAttr>(
          resultType.getEncoding());
  if (!specializerAttr || !specializerAttr.supportsSpecialization()) {
    return;
  }

  // Get the encoding_dims values
  auto encodingDims = setEncodingOp.getEncodingDims();
  if (encodingDims.empty()) {
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Creating markers for set_encoding: "
                          << *setEncodingOp << "\n");

  // Set insertion point before the set_encoding
  builder.setInsertionPoint(setEncodingOp);

  for (auto [idx, dim] : llvm::enumerate(encodingDims)) {
    // Skip constant values - only specialize on dynamic dimensions.
    // TODO: Remove this check once encoding_dims only contains dynamic dims.
    if (isConstantValue(dim)) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping constant encoding_dims[" << idx
                              << "]\n");
      continue;
    }

    auto key = std::make_pair(dim, static_cast<unsigned>(idx));

    // Check if we've already created markers for this (value, idx) pair
    if (seenKeys.contains(key)) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping duplicate for encoding_dims["
                              << idx << "]\n");
      continue;
    }
    seenKeys.insert(key);

    // Create util.specialize directly for the encoding_dims value
    auto specializeOp =
        IREE::Util::SpecializeOp::create(builder, setEncodingOp.getLoc(), dim);
    specializeOps.push_back(specializeOp);

    LLVM_DEBUG(llvm::dbgs() << "  Created util.specialize for encoding_dims["
                            << idx << "]: " << specializeOp.getResult()
                            << "\n");
  }
}

/// Check if `defOp` properly dominates `use` within the same region.
/// This is a simplified dominance check that works for operations in the same
/// block (defOp must come before use's owner) or different blocks within the
/// same region (we use a conservative approach - only replace if in same block
/// and defOp comes first).
static bool properlyDominates(Operation *defOp, OpOperand *use) {
  Operation *user = use->getOwner();

  // If in different regions, the outer region's op dominates inner uses
  if (defOp->getParentRegion() != user->getParentRegion()) {
    return defOp->getParentRegion()->isAncestor(user->getParentRegion());
  }

  // Same region - check if same block
  Block *defBlock = defOp->getBlock();
  Block *userBlock = user->getBlock();

  if (defBlock != userBlock) {
    // Different blocks in same region - for now, be conservative and don't
    // replace. A proper implementation would use DominanceInfo.
    return false;
  }

  // Same block - defOp must come before user in the block
  return defOp->isBeforeInBlock(user);
}

/// Get the nesting depth of a region (how many parent regions it has).
static unsigned getRegionNestingDepth(Region *region) {
  unsigned depth = 0;
  while (region) {
    ++depth;
    Operation *parentOp = region->getParentOp();
    region = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return depth;
}

/// After reification, the util.specialize ops now have their operands resolved
/// to the actual values (e.g., encoding_dims values). Replace uses of those
/// values that are dominated by the specialize op with the specialize results.
static void replaceUsesWithSpecializeResults(
    ArrayRef<IREE::Util::SpecializeOp> specializeOps) {
  // Map from (value, region) to their specialize ops
  // We need region-aware deduplication since the same value might need
  // different specialize ops in different dispatch regions
  DenseMap<std::pair<Value, Region *>, IREE::Util::SpecializeOp>
      valueToSpecialize;

  for (auto specOp : specializeOps) {
    Value operand = specOp.getOperand();
    Region *region = specOp->getParentRegion();
    auto key = std::make_pair(operand, region);

    // If we already have a specialize for this value in this region,
    // use that one and mark this one for removal
    auto it = valueToSpecialize.find(key);
    if (it != valueToSpecialize.end()) {
      // Replace uses of this specialize result with the earlier one
      specOp.getResult().replaceAllUsesWith(it->second.getResult());
      specOp.erase();
      continue;
    }

    valueToSpecialize[key] = specOp;
  }

  // Sort by region nesting depth (deepest first) to process inner specialize
  // ops before outer ones. This ensures that inner uses get replaced by inner
  // specialize results before the outer specialize can replace them.
  SmallVector<std::pair<std::pair<Value, Region *>, IREE::Util::SpecializeOp>>
      sortedEntries(valueToSpecialize.begin(), valueToSpecialize.end());
  llvm::sort(sortedEntries, [](const auto &a, const auto &b) {
    return getRegionNestingDepth(a.first.second) >
           getRegionNestingDepth(b.first.second);
  });

  // Now replace uses of the original values with the specialize results
  // Only replace uses that are dominated by the specialize op
  // Process from innermost to outermost regions so inner specialize ops
  // get their uses replaced before outer ones can interfere.
  for (auto &[key, specOp] : sortedEntries) {
    Value value = key.first;

    LLVM_DEBUG(llvm::dbgs() << "Processing specialize for value " << value
                            << " in region\n");

    // Collect uses to replace (can't modify while iterating)
    SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      // Skip the specialize op itself
      if (user == specOp.getOperation()) {
        continue;
      }
      // Only replace uses that are properly dominated by the specialize op
      if (properlyDominates(specOp.getOperation(), &use)) {
        usesToReplace.push_back(&use);
      }
    }

    // Replace the collected uses
    for (OpOperand *use : usesToReplace) {
      use->set(specOp.getResult());
      LLVM_DEBUG(llvm::dbgs() << "  Replaced use in: " << *use->getOwner()
                              << "\n");
    }
  }
}

struct InsertEncodingSpecializationMarkersPass
    : public impl::InsertEncodingSpecializationMarkersPassBase<
          InsertEncodingSpecializationMarkersPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // Track which (operand, dimIndex) pairs we've already processed
    DenseSet<std::pair<Value, unsigned>> seenKeys;
    // Collect all created specialize ops
    SmallVector<IREE::Util::SpecializeOp> specializeOps;

    // First, handle set_encoding ops - insert specialize markers for their
    // encoding_dims before any other processing
    funcOp.walk([&](IREE::Encoding::SetEncodingOp setEncodingOp) {
      insertMarkersForSetEncoding(setEncodingOp, builder, seenKeys,
                                  specializeOps);
    });

    // Walk all dispatch regions and create dim + specialize ops for ops
    // that use encoded tensors
    funcOp.walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      regionOp.walk([&](Operation *op) {
        // Skip the dispatch region op itself and flow.return
        if (op == regionOp.getOperation() || isa<IREE::Flow::ReturnOp>(op)) {
          return;
        }

        insertMarkersForOp(op, builder, seenKeys, specializeOps);
      });
    });

    // If no markers were created, nothing to do
    if (specializeOps.empty()) {
      return;
    }

    // Apply reification patterns to fold encoding_dim ops at function level.
    // This traces iree_encoding.dim ops back through the producer chain
    // to resolve them to actual values (e.g., encoding_dims from set_encoding).
    // The util.specialize ops will have their operands updated to the resolved
    // values.
    RewritePatternSet patterns(context);
    IREE::Encoding::populateDimReificationPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    // After reification, replace uses of the resolved values with the
    // specialize results. This ensures downstream passes see the specialized
    // values.
    replaceUsesWithSpecializeResults(specializeOps);
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
