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
#include "llvm/ADT/SetVector.h"
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

/// After reification, the util.specialize ops now have their operands resolved
/// to the actual values (e.g., encoding_dims values). Replace uses of those
/// values that are dominated by the specialize op with the specialize results.
static void replaceUsesWithSpecializeResults(
    ArrayRef<IREE::Util::SpecializeOp> specializeOps) {
  for (auto specOp : specializeOps) {
    Value operand = specOp.getOperand();

    // Collect uses to replace (can't modify while iterating)
    SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : operand.getUses()) {
      Operation *user = use.getOwner();
      // Skip the specialize op itself
      if (user == specOp.getOperation()) {
        continue;
      }
      // Only replace uses that come after the specialize op in the same block
      if (user->getBlock() == specOp->getBlock() &&
          specOp->isBeforeInBlock(user)) {
        usesToReplace.push_back(&use);
      }
    }

    // Replace the collected uses
    for (OpOperand *use : usesToReplace) {
      use->set(specOp.getResult());
      LLVM_DEBUG(llvm::dbgs() << "Replaced use in: " << *use->getOwner()
                              << "\n");
    }
  }
}

/// Creates iree_encoding.dim ops wrapped with util.specialize for the 
/// dimensions that need specialization. Returns the created specialize ops.
static void createDimAndSpecializeOpsForOp(
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

    LLVM_DEBUG(llvm::dbgs() << "Creating dim+specialize ops for op: " << *op
                            << "\n");

    // Set insertion point BEFORE the operation
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

      // Wrap with util.specialize as an anchor that survives reification
      auto specializeOp = IREE::Util::SpecializeOp::create(
          builder, op->getLoc(), encodingDimOp.getResult());
      specializeOps.push_back(specializeOp);

      LLVM_DEBUG(llvm::dbgs() << "  Created encoding_dim + specialize for dim "
                              << specOp.dimIndex << "\n");
    }
  }
}

/// Processes a dispatch region: creates dim ops wrapped with specialize ops,
/// reifies them, and collects the resolved values to add as
/// specialization_values to the dispatch region.
static void processDispatchRegion(IREE::Flow::DispatchRegionOp regionOp,
                                  OpBuilder &builder, MLIRContext *context) {
  // Track which (operand, dimIndex) pairs we've already processed
  DenseSet<std::pair<Value, unsigned>> seenKeys;
  // Collect all created specialize ops (these survive reification)
  SmallVector<IREE::Util::SpecializeOp> specializeOps;

  // Create dim + specialize ops for all ops inside the dispatch region
  regionOp.walk([&](Operation *op) {
    // Skip the dispatch region op itself and flow.return
    if (op == regionOp.getOperation() || isa<IREE::Flow::ReturnOp>(op)) {
      return;
    }
    createDimAndSpecializeOpsForOp(op, builder, seenKeys, specializeOps);
  });

  if (specializeOps.empty()) {
    return;
  }

  // Apply reification patterns to fold encoding_dim ops.
  // The specialize ops serve as anchors - after reification, their operands
  // will be the resolved values (e.g., encoding_dims from set_encoding).
  RewritePatternSet patterns(context);
  IREE::Encoding::populateDimReificationPatterns(patterns);

  // Apply to the function containing this dispatch region
  auto funcOp = regionOp->getParentOfType<mlir::FunctionOpInterface>();
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to apply reification patterns\n");
    // Clean up specialize ops
    for (auto specOp : specializeOps) {
      specOp.getResult().replaceAllUsesWith(specOp.getOperand());
      specOp.erase();
    }
    return;
  }

  // Collect the resolved values from the specialize ops' operands.
  // After reification, the operands of specialize ops are the resolved values.
  // SetVector automatically deduplicates.
  llvm::SetVector<Value> specializationValues;
  for (auto specOp : specializeOps) {
    Value resolvedValue = specOp.getOperand();
    // Only add if it's defined outside the dispatch region
    // (i.e., it's a value we need to capture)
    if (auto *defOp = resolvedValue.getDefiningOp()) {
      if (!regionOp->isAncestor(defOp)) {
        specializationValues.insert(resolvedValue);
      }
    } else {
      // Block argument - include it
      specializationValues.insert(resolvedValue);
    }
  }

  // Clean up the specialize ops - they served their purpose as anchors
  for (auto specOp : specializeOps) {
    specOp.getResult().replaceAllUsesWith(specOp.getOperand());
    specOp.erase();
  }

  if (specializationValues.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No specialization values found after "
                               "reification for dispatch region\n");
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Specialization values for dispatch region:\n";
    for (Value v : specializationValues) {
      llvm::dbgs() << "  " << v << "\n";
    }
  });

  // Update the dispatch region to include the specialization values
  // We need to create a new dispatch region with the additional operands
  builder.setInsertionPoint(regionOp);

  // Get the current operands
  SmallVector<Value> resultDims(regionOp.getResultDims());
  SmallVector<Value> workload(regionOp.getWorkload());
  SmallVector<Value> specValues(specializationValues.begin(),
                                specializationValues.end());

  // Create a new dispatch region with specialization values
  auto newRegionOp = IREE::Flow::DispatchRegionOp::create(
      builder, regionOp.getLoc(), regionOp.getResultTypes(), resultDims,
      workload, specValues);

  // Move the body and workgroup_count regions
  newRegionOp.getBody().takeBody(regionOp.getBody());
  if (!regionOp.getWorkgroupCount().empty()) {
    newRegionOp.getWorkgroupCount().takeBody(regionOp.getWorkgroupCount());
  }

  // Replace uses and erase the old op
  regionOp.replaceAllUsesWith(newRegionOp.getResults());
  regionOp.erase();
}

struct InsertEncodingSpecializationMarkersPass
    : public impl::InsertEncodingSpecializationMarkersPassBase<
          InsertEncodingSpecializationMarkersPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    // First, process dispatch regions - add specialization values to them.
    // This must happen before set_encoding handling so the dispatch regions
    // see clean values without util.specialize wrappers.
    SmallVector<IREE::Flow::DispatchRegionOp> regionOps;
    funcOp.walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      regionOps.push_back(regionOp);
    });

    for (auto regionOp : regionOps) {
      processDispatchRegion(regionOp, builder, context);
    }

    // Then, handle set_encoding ops - insert util.specialize markers for their
    // encoding_dims. These stay as util.specialize because set_encoding ops
    // are outside dispatch regions and will be picked up by MaterializeEncodings.
    DenseSet<std::pair<Value, unsigned>> seenKeys;
    SmallVector<IREE::Util::SpecializeOp> specializeOps;

    funcOp.walk([&](IREE::Encoding::SetEncodingOp setEncodingOp) {
      insertMarkersForSetEncoding(setEncodingOp, builder, seenKeys,
                                  specializeOps);
    });

    // Apply reification for set_encoding markers and replace uses
    if (!specializeOps.empty()) {
      RewritePatternSet patterns(context);
      IREE::Encoding::populateDimReificationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
      replaceUsesWithSpecializeResults(specializeOps);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
