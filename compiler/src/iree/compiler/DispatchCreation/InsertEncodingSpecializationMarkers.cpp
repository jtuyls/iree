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
#include "mlir/IR/Dominance.h"
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

/// Process a set_encoding op: create util.specialize ops for encoding dims
/// that need specialization and update the set_encoding to use them.
static void processSetEncodingOp(IREE::Encoding::SetEncodingOp setEncodingOp,
                                 DenseMap<Value, Value> &specializeCache) {
  auto resultType = setEncodingOp.getResultType();
  auto encoding = resultType.getEncoding();
  if (!encoding) {
    return;
  }

  auto specializerAttr =
      dyn_cast<IREE::Encoding::DynamicLayoutSpecializerAttr>(encoding);
  if (!specializerAttr || !specializerAttr.supportsSpecialization()) {
    return;
  }

  ValueRange encodingDims = setEncodingOp.getEncodingDims();
  if (encodingDims.empty()) {
    return;
  }

  SmallVector<unsigned> specDimIndices =
      specializerAttr.getSpecializationDimensions();
  if (specDimIndices.empty()) {
    for (unsigned i = 0; i < encodingDims.size(); ++i) {
      specDimIndices.push_back(i);
    }
  }

  OpBuilder builder(setEncodingOp);
  SmallVector<Value> newEncodingDims(encodingDims.begin(), encodingDims.end());
  bool modified = false;

  for (unsigned dimIdx : specDimIndices) {
    if (dimIdx >= encodingDims.size()) {
      continue;
    }

    Value dimValue = encodingDims[dimIdx];

    // Check if we've already created a specialize op for this value
    Value specValue;
    auto it = specializeCache.find(dimValue);
    if (it != specializeCache.end()) {
      specValue = it->second;
    } else {
      // Create the specialize op before the set_encoding
      auto specOp = IREE::Util::SpecializeOp::create(builder,
                                                      setEncodingOp.getLoc(),
                                                      dimValue);
      specValue = specOp.getResult();
      specializeCache[dimValue] = specValue;

      LLVM_DEBUG(llvm::dbgs() << "Created util.specialize: " << dimValue
                              << " -> " << specValue << "\n");
    }

    newEncodingDims[dimIdx] = specValue;
    modified = true;
  }

  if (modified) {
    setEncodingOp.getEncodingDimsMutable().assign(newEncodingDims);
    LLVM_DEBUG(llvm::dbgs() << "Updated set_encoding: " << setEncodingOp
                            << "\n");
  }
}

/// Hoist util.specialize ops into dispatch regions.
/// For each dispatch region:
/// 1. Find values used in the region that come from util.specialize ops
///    outside and create new specialize ops inside
/// 2. Find values used in the region that are the INPUT to a util.specialize
///    op outside (meaning they need specialization) and create specialize
///    ops for them inside the dispatch
static void hoistSpecializeOpsIntoDispatchRegions(
    FunctionOpInterface funcOp,
    const DenseMap<Value, Value> &specializeCache) {
  funcOp.walk([&](IREE::Flow::DispatchRegionOp regionOp) {
    Region &body = regionOp.getBody();
    if (body.empty()) {
      return;
    }

    Block &entryBlock = body.front();

    // First pass: collect all values that need specialization inside the
    // dispatch. We collect them first to avoid modifying the IR while walking.
    DenseSet<Value> valuesToSpecialize;

    regionOp.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        // Skip values already marked
        if (valuesToSpecialize.contains(operand)) {
          continue;
        }

        // Skip values defined inside the region
        if (operand.getDefiningOp() &&
            operand.getDefiningOp()->getParentRegion() == &body) {
          continue;
        }

        // Case 1: Check if this operand is defined by a util.specialize op
        // outside the region - we'll create a new one inside
        if (operand.getDefiningOp<IREE::Util::SpecializeOp>()) {
          valuesToSpecialize.insert(operand);
          continue;
        }

        // Case 2: Check if this operand is the INPUT to a util.specialize op
        // (meaning it should be specialized when captured)
        if (specializeCache.contains(operand)) {
          valuesToSpecialize.insert(operand);
        }
      }
    });

    if (valuesToSpecialize.empty()) {
      return;
    }

    // Second pass: create specialize ops at the beginning of the region
    OpBuilder builder(&entryBlock, entryBlock.begin());
    DenseMap<Value, Value> specValueMapping;

    for (Value value : valuesToSpecialize) {
      // Get the original (unspecialized) value to use as input
      Value inputValue = value;
      if (auto specOp = value.getDefiningOp<IREE::Util::SpecializeOp>()) {
        inputValue = specOp.getOperand();
      }

      auto newSpecOp = IREE::Util::SpecializeOp::create(
          builder, value.getLoc(), inputValue);
      specValueMapping[value] = newSpecOp.getResult();

      LLVM_DEBUG(llvm::dbgs()
                 << "Created util.specialize inside dispatch: " << inputValue
                 << " -> " << newSpecOp.getResult() << "\n");
    }

    // Third pass: replace uses of the outer values with the specialized ones
    for (auto &[outerValue, innerValue] : specValueMapping) {
      outerValue.replaceUsesWithIf(innerValue, [&](OpOperand &use) {
        // Don't replace in the specialize op itself
        if (use.getOwner() == innerValue.getDefiningOp()) {
          return false;
        }
        return use.getOwner()->getParentRegion() == &body;
      });
    }
  });
}

/// Remove util.specialize ops that have no uses (after hoisting into
/// dispatches, the outer specialize ops may become unused).
static void removeUnusedSpecializeOps(FunctionOpInterface funcOp) {
  SmallVector<IREE::Util::SpecializeOp> opsToRemove;
  funcOp.walk([&](IREE::Util::SpecializeOp specOp) {
    if (specOp.getResult().use_empty()) {
      opsToRemove.push_back(specOp);
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

    // Track values we've already specialized to avoid duplicates
    DenseMap<Value, Value> specializeCache;

    // Collect set_encoding ops first to avoid iterator invalidation
    SmallVector<IREE::Encoding::SetEncodingOp> setEncodingOps;
    funcOp.walk([&](IREE::Encoding::SetEncodingOp op) {
      setEncodingOps.push_back(op);
    });

    if (setEncodingOps.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No set_encoding ops found\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Processing " << setEncodingOps.size()
                            << " set_encoding ops\n");

    // First pass: create specialize ops and update set_encoding ops
    for (auto setEncodingOp : setEncodingOps) {
      processSetEncodingOp(setEncodingOp, specializeCache);
    }

    // Second pass: hoist specialize ops into dispatch regions
    hoistSpecializeOpsIntoDispatchRegions(funcOp, specializeCache);

    // Third pass: remove unused specialize ops
    removeUnusedSpecializeOps(funcOp);
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
