// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::DispatchCreation {

#define DEBUG_TYPE "iree-dispatch-creation-insert-encoding-specialization-markers"

#define GEN_PASS_DEF_INSERTENCODINGSPECIALIZATIONMARKERSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Inserts util.specialize markers for a tensor with a specializable encoding.
/// Returns the encoding_dim ops created so they can be used as workload values.
static SmallVector<Value> insertMarkersForTensor(
    OpBuilder &builder, Location loc, Value tensor,
    IREE::Encoding::DynamicLayoutSpecializerAttr specializerAttr) {
  auto specDims = specializerAttr.getSpecializationDimensions();
  if (specDims.empty()) {
    return {};
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserting specialization markers for tensor: "
                          << tensor << "\n");

  SmallVector<Value> encodingDimValues;
  for (unsigned dimIndex : specDims) {
    // Create the encoding_dim op to query this dimension
    auto encodingDimOp = IREE::Encoding::EncodingDimOp::create(
        builder, loc, tensor, static_cast<int64_t>(dimIndex));
    encodingDimValues.push_back(encodingDimOp.getResult());

    // Create the specialize op to mark this value for specialization.
    // The actual ranges are stored in the encoding attribute and will be
    // retrieved by SpecializeExports from SpecializableLayoutAttr.
    IREE::Util::SpecializeOp::create(builder, loc, encodingDimOp.getResult());

    LLVM_DEBUG(llvm::dbgs() << "  Created marker for dim " << dimIndex << "\n");
  }

  return encodingDimValues;
}

/// Extracts DynamicLayoutSpecializerAttr from a tensor type's encoding.
static IREE::Encoding::DynamicLayoutSpecializerAttr
getSpecializerFromType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType || !tensorType.getEncoding()) {
    return nullptr;
  }
  return dyn_cast<IREE::Encoding::DynamicLayoutSpecializerAttr>(
      tensorType.getEncoding());
}

/// Walks a dispatch region looking for tensors with specializable encodings
/// and inserts markers for them.
static void processDispatchRegion(IREE::Flow::DispatchRegionOp regionOp) {
  OpBuilder builder(regionOp.getContext());

  // Walk operations inside the dispatch region looking for tensors with
  // specializable encodings
  regionOp.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      auto specializerAttr = getSpecializerFromType(result.getType());
      if (!specializerAttr || !specializerAttr.supportsSpecialization()) {
        continue;
      }

      // Set insertion point after the op that produces the tensor
      builder.setInsertionPointAfter(op);
      insertMarkersForTensor(builder, op->getLoc(), result, specializerAttr);
    }
  });
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
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation

