// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/EncodingExternalModels.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "iree-encoding-external-models"

namespace mlir::iree_compiler {
namespace {

struct ContractionAttrPropagationInterface
    : public IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
          ContractionAttrPropagationInterface, IREE::Encoding::MatmulKAttr> {
  bool isPropagable(Attribute attr, Value target) const {
    auto encoding = cast<IREE::Encoding::MatmulKAttr>(attr);
    Operation *attachedToOperation = target.getDefiningOp();
    if (!attachedToOperation) {
      return false;
    }
    return TypeSwitch<Operation *, bool>(attachedToOperation)
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          ArrayRef<int32_t> kDims = encoding.getKDims().asArrayRef();
          // TODO: Relax the check to allow transforming innermost reduction
          // dimensions. We need to revisit the matmul_k encoding semantic.
          SmallVector<ReassociationIndices, 4> reassociationMaps =
              collapseOp.getReassociationIndices();
          for (int32_t k : kDims) {
            if (reassociationMaps[k].size() != 1) {
              return false;
            }
          }
          return true;
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateEncodings(Attribute attr, Value target) const {
    auto encoding = cast<IREE::Encoding::MatmulKAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target.getDefiningOp())
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          ArrayRef<int32_t> kDims = encoding.getKDims().asArrayRef();
          SmallVector<ReassociationIndices, 4> reassociationMaps =
              collapseOp.getReassociationIndices();
          // Get a mapping from original iteration space to expanded iteration
          // space.
          SmallVector<int32_t> newKDims;
          for (int32_t kDim : kDims) {
            newKDims.append(reassociationMaps[kDim].begin(),
                            reassociationMaps[kDim].end());
          }
          MLIRContext *ctx = collapseOp.getContext();
          auto operandEncodingAttr =
              IREE::Encoding::MatmulKAttr::get(ctx, newKDims);
          IREE::Encoding::PropagationEncoding propEncoding;
          propEncoding.operandEncodings.push_back(operandEncodingAttr);
          // The result encoding will be the same as the encoding
          // present in the set encoding operation.
          propEncoding.resultEncodings.push_back(encoding);
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct ContractionOpPropagationInterface
    : public IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
          ContractionOpPropagationInterface, tensor::CollapseShapeOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &builder,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    Location loc = op->getLoc();
    auto operandEncodings = encodings.operandEncodings;
    auto resultEncodings = encodings.resultEncodings;
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<tensor::CollapseShapeOp>([&](auto collapseOp) {
          RankedTensorType operandEncodingType =
              collapseOp.getSrcType().cloneWithEncoding(
                  operandEncodings.front());
          Value newEncodingOp = builder.create<IREE::Encoding::SetEncodingOp>(
              loc, operandEncodingType, collapseOp.getSrc());
          auto resultEncodingType =
              dyn_cast<RankedTensorType>(opResult.getType())
                  .cloneWithEncoding(resultEncodings.front());
          Value newCollapseOp = builder.create<tensor::CollapseShapeOp>(
              loc, resultEncodingType, newEncodingOp,
              collapseOp.getReassociationIndices());
          IREE::Encoding::PropagationResult result;
          result.replacement = newCollapseOp;
          result.generatedEncodingOps.push_back(newEncodingOp.getDefiningOp());
          return result;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct EncodingAttrPropagationInterface
    : public IREE::Encoding::EncodingPropagationAttrInterface::ExternalModel<
    EncodingAttrPropagationInterface, IREE::Encoding::EncodingAttr> {
  bool isPropagable(Attribute attr, Value target) const {
    auto encoding = cast<IREE::Encoding::EncodingAttr>(attr);
    Operation *attachedToOperation = target.getDefiningOp();
    if (!attachedToOperation) {
      return false;
    }
    return TypeSwitch<Operation *, bool>(attachedToOperation)
        .Case<linalg::GenericOp>([&](auto genericOp) {
          if (genericOp.getNumReductionLoops() != 0) {
            return false;
          }
          return true;
        })
        .Default([&](auto) { return false; });
  }

  FailureOr<IREE::Encoding::PropagationEncoding>
  generateEncodings(Attribute attr, Value target) const {
    auto encoding = cast<IREE::Encoding::EncodingAttr>(attr);
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationEncoding>>(
               target.getDefiningOp())
        .Case<linalg::GenericOp>([&](auto genericOp) {
          IREE::Encoding::PropagationEncoding propEncoding;
          for (OpOperand *operand : genericOp.getDpsInputOperands()) {
            (void)operand;
            propEncoding.operandEncodings.push_back(encoding);
          }
          // The result encoding will be the same as the encoding
          // present in the set encoding operation.
          propEncoding.resultEncodings.push_back(encoding);
          return propEncoding;
        })
        .Default([&](auto) { return failure(); });
  }
};

struct GenericOpPropagationInterface
    : public IREE::Encoding::EncodingPropagationOpInterface::ExternalModel<
    GenericOpPropagationInterface, linalg::GenericOp> {
  FailureOr<IREE::Encoding::PropagationResult>
  propagateEncoding(Operation *op, RewriterBase &rewriter,
                    IREE::Encoding::PropagationEncoding encodings,
                    OpResult opResult) const {
    auto genericOp = cast<linalg::GenericOp>(op);
    LLVM_DEBUG(llvm::dbgs() << "opResult.getOwner(): " << *opResult.getOwner() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "op: " << genericOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "value: " << cast<Value>(opResult) << "\n");
    Location loc = op->getLoc();
    auto operandEncodings = encodings.operandEncodings;
    auto resultEncodings = encodings.resultEncodings;
    return TypeSwitch<Operation *,
                      FailureOr<IREE::Encoding::PropagationResult>>(
               opResult.getOwner())
        .Case<IREE::Encoding::UnsetEncodingOp>([&](auto encodingOp) -> FailureOr<IREE::Encoding::PropagationResult> {
          LLVM_DEBUG(llvm::dbgs() << "linalg::GenericOp\n");
          IREE::Encoding::PropagationResult result;
          // Set encodings on each input
          Location loc = genericOp->getLoc();
          SmallVector<Value> encodedOperands;
          for (auto &&[operand, encoding] : llvm::zip_equal(genericOp.getDpsInputOperands(), operandEncodings)) {
            auto encodingAttr = cast<IREE::Encoding::EncodingAttr>(encoding);
            // Append the operand's indexing map to the encoding's user indexing maps.
            AffineMap operandMap = genericOp.getMatchingIndexingMap(operand);

            // SmallVector<Value> resultDims;
            Operation *sourceOp = operand->get().getDefiningOp();
            if (sourceOp && sourceOp == encodingOp) {
              encodedOperands.push_back(encodingOp.getSource());
              continue;
            }
            // LLVM_DEBUG(llvm::dbgs() << "operand: " << *operand->get().getDefiningOp() << "\n");

            // Create new encoding and set encoding on the operand.
            IREE::Encoding::EncodingAttr newEncoding =
              encodingAttr.cloneWithNewOperandIndexingMap(operandMap);
            auto operandType = cast<RankedTensorType>(operand->get().getType());
            auto resType = RankedTensorType::get(
                operandType.getShape(), operandType.getElementType(), newEncoding);
            Value encodedInput = rewriter.create<IREE::Encoding::SetEncodingOp>(
                loc, resType, operand->get());
            result.generatedEncodingOps.push_back(encodedInput.getDefiningOp());
            encodedOperands.push_back(encodedInput);
            }

          auto emptyOp = genericOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
          if (!emptyOp) {
            return failure();
          }
          auto resultEncodingType =
              dyn_cast<RankedTensorType>(emptyOp.getResult().getType())
                  .cloneWithEncoding(resultEncodings.front());

          // Create encoded generic op.
          // SmallVector<OpFoldResult> mixedSizes =
          // tensor::getMixedSizes(rewriter, loc, encodingOp.getSource());
          // Value encodedInit = rewriter.create<tensor::EmptyOp>(
          //   loc, mixedSizes, resultEncodingType.getElementType(), resultEncodings.front());
          Value encodedInit;
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointAfter(emptyOp);
            encodedInit = rewriter.create<tensor::EmptyOp>(
              loc, emptyOp.getType().getShape(),
              resultEncodingType.getElementType(),
              emptyOp.getDynamicSizes(), resultEncodings.front());
          }
          // auto encodedInit =  rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
          //   emptyOp, emptyOp.getType().getShape(),
          //   resultEncodingType.getElementType(),
          //   emptyOp.getDynamicSizes(), resultEncodings.front());
          encodedOperands.push_back(encodedInit);
          rewriter.setInsertionPointAfter(genericOp);
          auto encodedGenericOp =
            clone(rewriter, genericOp, resultEncodingType, encodedOperands);
          
          LLVM_DEBUG(llvm::dbgs() << "RESULT\n");
          result.replacement = encodedGenericOp.getResult(0);
          // result.generatedEncodingOps.push_back(newEncodingOp.getDefiningOp());
          return result;
        })
        .Default([&](auto) { return failure(); });
  }
};

} // namespace

void registerEncodingExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Encoding::IREEEncodingDialect *dialect) {
        IREE::Encoding::MatmulKAttr::attachInterface<
            ContractionAttrPropagationInterface>(*ctx);
        IREE::Encoding::EncodingAttr::attachInterface<
            EncodingAttrPropagationInterface>(*ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx,
                            mlir::tensor::TensorDialect *dialect) {
    tensor::CollapseShapeOp::attachInterface<ContractionOpPropagationInterface>(
        *ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx,
    mlir::linalg::LinalgDialect *dialect) {
      linalg::GenericOp::attachInterface<GenericOpPropagationInterface>(
      *ctx);
  });
}

} // namespace mlir::iree_compiler
