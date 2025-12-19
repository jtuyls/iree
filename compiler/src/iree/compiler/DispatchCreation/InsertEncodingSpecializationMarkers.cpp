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

/// Walks a dispatch region looking for operations with specializable encodings
/// and inserts markers for them using the getSpecializationOperands API.
static void processDispatchRegion(IREE::Flow::DispatchRegionOp regionOp) {
  OpBuilder builder(regionOp.getContext());

  // Track (operand, dimIndex) pairs we've already processed to avoid
  // creating duplicate iree_encoding.encoding_dim ops.
  DenseMap<std::pair<Value, unsigned>, Value> encodingDimCache;

  // Walk operations inside the dispatch region
  regionOp.walk([&](Operation *op) {
    // Skip flow.return and other non-payload ops
    if (isa<IREE::Flow::ReturnOp>(op)) {
      return;
    }

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

      LLVM_DEBUG(llvm::dbgs()
                 << "Inserting specialization markers for op: " << *op << "\n");

      // Set insertion point after the operation
      builder.setInsertionPointAfter(op);

      for (const auto &specOp : specOperands) {
        auto key = std::make_pair(specOp.operand, specOp.dimIndex);

        // Check if we've already created an encoding_dim for this
        // (operand, dimIndex) pair
        Value encodingDimValue;
        auto it = encodingDimCache.find(key);
        if (it != encodingDimCache.end()) {
          encodingDimValue = it->second;
          LLVM_DEBUG(llvm::dbgs()
                     << "  Reusing encoding_dim for operand dim "
                     << specOp.dimIndex << "\n");
        } else {
          // Create the encoding_dim op
          auto encodingDimOp = IREE::Encoding::EncodingDimOp::create(
              builder, op->getLoc(), specOp.operand,
              static_cast<int64_t>(specOp.dimIndex));
          encodingDimValue = encodingDimOp.getResult();
          encodingDimCache[key] = encodingDimValue;

          LLVM_DEBUG(llvm::dbgs()
                     << "  Created encoding_dim for operand dim "
                     << specOp.dimIndex << ": " << encodingDimValue << "\n");
        }

        // Create the specialize op
        IREE::Util::SpecializeOp::create(builder, op->getLoc(),
                                         encodingDimValue);

        LLVM_DEBUG(llvm::dbgs()
                   << "  Created util.specialize for dim " << specOp.dimIndex
                   << "\n");
      }
    }
  });
}

/// Removes duplicate util.specialize ops within a dispatch region.
/// After encoding_dim ops are canonicalized, multiple util.specialize ops
/// may end up consuming the same SSA value.
static void removeDuplicateSpecializeOps(IREE::Flow::DispatchRegionOp regionOp) {
  DenseSet<Value> specializedValues;
  SmallVector<IREE::Util::SpecializeOp> opsToRemove;

  regionOp.walk([&](IREE::Util::SpecializeOp specOp) {
    Value operand = specOp.getOperand();
    if (specializedValues.contains(operand)) {
      // This value has already been specialized, mark for removal
      opsToRemove.push_back(specOp);
      LLVM_DEBUG(llvm::dbgs() << "Removing duplicate util.specialize for: "
                              << operand << "\n");
    } else {
      specializedValues.insert(operand);
    }
  });

  for (auto op : opsToRemove) {
    op.erase();
  }
}

struct InsertEncodingSpecializationMarkersPass
    : public impl::InsertEncodingSpecializationMarkersPassBase<
          InsertEncodingSpecializationMarkersPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Walk all dispatch regions in the function
    funcOp.walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      processDispatchRegion(regionOp);
    });

    // Apply reification patterns to fold encoding_dim ops to tensor.dim
    RewritePatternSet patterns(&getContext());
    IREE::Encoding::populateEncodingDimReificationPatterns(patterns);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove duplicate util.specialize ops after canonicalization
    funcOp.walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      removeDuplicateSpecializeOps(regionOp);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
