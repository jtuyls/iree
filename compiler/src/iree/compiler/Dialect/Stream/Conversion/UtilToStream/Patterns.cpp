// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/UtilToStream/Patterns.h"

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

struct FuncOpSignatureConversion
    : public OpConversionPattern<IREE::Util::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Replace function and convert the signature for region conversion below.
    TypeConverter::SignatureConversion newSignature(funcOp.getNumArguments());
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    bool anyFailed = false;
    newFuncOp.expandSignature(
        [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
          if (failed(typeConverter.convertTypes(type, newTypes))) {
            anyFailed = true;
          }
          if (failed(
                  typeConverter.convertSignatureArg(i, type, newSignature))) {
            anyFailed = true;
          }
        },
        [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
          if (failed(typeConverter.convertTypes(type, newTypes))) {
            anyFailed = true;
          }
        });
    if (anyFailed) {
      return rewriter.notifyMatchFailure(
          funcOp, "unable to convert argument/result types");
    }
    newFuncOp.getBlocks().clear();
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter, &newSignature))) {
      return failure();
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<IREE::Util::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original call as
    // the result counts differ.
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    bool anyFailed = false;
    auto callOp = op.cloneAndExpand(
        [&](unsigned i, Value operand, SmallVectorImpl<Value> &newOperands) {
          auto adaptorOperand = adaptor.getOperands()[i];
          expandResourceOperand(op.getLoc(), adaptorOperand, newOperands,
                                rewriter);
        },
        [&](unsigned i, Type type, SmallVectorImpl<Type> &newTypes) {
          size_t newIndex = newTypes.size();
          if (failed(getTypeConverter()->convertType(type, newTypes)))
            anyFailed = true;
          resultMap.push_back(Result{i, newIndex, newTypes[newIndex]});
        },
        rewriter);
    if (anyFailed) {
      return rewriter.notifyMatchFailure(op, "unable to convert result types");
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = callOp.getResult(result.newIndex + 0);
        auto resourceSize = callOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(callOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<IREE::Util::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(op, expandedOperands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

struct ExpandedGlobalResource {
  IREE::Util::GlobalOp resourceOp;
  IREE::Util::GlobalOp resourceSizeOp;
};

struct GlobalExpansionState {
  // A map of original symbol name to one new global for each expanded type.
  DenseMap<StringRef, ExpandedGlobalResource> globalMap;
};

static bool isExpandedType(Type type) {
  if (llvm::isa<TensorType>(type))
    return true;
  if (auto ptrType = llvm::dyn_cast<IREE::Util::PtrType>(type)) {
    return isExpandedType(ptrType);
  }
  return false;
}

template <typename T>
class BaseGlobalConversionPattern : public OpConversionPattern<T> {
public:
  BaseGlobalConversionPattern(
      std::shared_ptr<GlobalExpansionState> expansionState,
      TypeConverter &typeConverter, MLIRContext *context,
      PatternBenefit benefit = 1)
      : OpConversionPattern<T>(typeConverter, context, benefit),
        expansionState(std::move(expansionState)) {}

protected:
  mutable std::shared_ptr<GlobalExpansionState> expansionState;
};

struct GlobalOpExpansion
    : public BaseGlobalConversionPattern<IREE::Util::GlobalOp> {
  using BaseGlobalConversionPattern::BaseGlobalConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only apply to expanded types (tensors/etc).
    if (!isExpandedType(globalOp.getType()))
      return failure();

    SmallVector<Type> newTypes;
    if (failed(getTypeConverter()->convertType(globalOp.getType(), newTypes))) {
      return rewriter.notifyMatchFailure(globalOp,
                                         "failed to convert ptr type");
    }
    if (newTypes.size() == 1 && newTypes.front() == globalOp.getType()) {
      return rewriter.notifyMatchFailure(globalOp, "no conversion needed");
    }

    // Start with the appropriate type. Lifetime refinement will use this as a
    // seed. Note that what was a constant in earlier dialects becomes a mutable
    // global holding a resource that may have constant contents.
    bool hasConstantUsage = !globalOp.getIsMutable();
    auto resourceType = IREE::Stream::ResourceType::get(
        rewriter.getContext(), hasConstantUsage
                                   ? IREE::Stream::Lifetime::Constant
                                   : IREE::Stream::Lifetime::Variable);

    // Special handling of the initial value: if it's a tensor then we need to
    // materialize an initializer and initialization ops. This allows the
    // current conversion to pick up the expanded initialization ops.
    auto initialValueAttr = globalOp.getInitialValueAttr();
    bool tensorInitializerRequired =
        initialValueAttr ? llvm::isa<TensorType>(initialValueAttr.getType())
                         : false;

    // New global holding the initial value only if it is not a tensor type.
    auto resourceOp = rewriter.replaceOpWithNewOp<IREE::Util::GlobalOp>(
        globalOp, globalOp.getName(), globalOp.getIsMutable(), resourceType,
        initialValueAttr && !tensorInitializerRequired
            ? std::optional<TypedAttr>{initialValueAttr}
            : std::nullopt);
    resourceOp.setVisibility(globalOp.getVisibility());

    // NOTE: we ignore noinline here, possibly to our peril. In earlier dialects
    // noinline indicates that the constant value should not be inlined, while
    // here it would be indicating the reference to the constant value should
    // not be (and that's weird).

    // Also create a global for tracking the resource size. In many cases this
    // is constant and will fold throughout the program. Global optimizations
    // such as same-value deduplication will also take effect.
    auto indexType = rewriter.getIndexType();
    auto resourceSizeOp = rewriter.create<IREE::Util::GlobalOp>(
        globalOp.getLoc(), (globalOp.getName() + "__size").str(),
        globalOp.getIsMutable(), indexType, std::optional<TypedAttr>{});
    resourceSizeOp.setVisibility(globalOp.getVisibility());

    // Materialize the initializer if we need to setup a tensor-like constant.
    if (tensorInitializerRequired) {
      auto affinityAttr = IREE::Stream::AffinityAttr::lookup(globalOp);
      auto initializerOp =
          rewriter.create<IREE::Util::InitializerOp>(globalOp.getLoc());
      auto *entryBlock = rewriter.createBlock(&initializerOp.getBody());
      rewriter.setInsertionPointToStart(entryBlock);
      Value initialValue, initialValueSize;
      if (initialValueAttr.isa<IREE::Util::UninitializedAttr>()) {
        initialValueSize = rewriter.create<IREE::Stream::TensorSizeOfOp>(
            globalOp.getLoc(), TypeAttr::get(globalOp.getType()),
            /*result_encoding_dims=*/ValueRange{}, affinityAttr);
        initialValue = rewriter.create<IREE::Stream::TensorEmptyOp>(
            globalOp.getLoc(), resourceOp.getType(),
            TypeAttr::get(globalOp.getType()),
            /*result_encoding_dims=*/ValueRange{}, initialValueSize,
            affinityAttr);
      } else {
        initialValue = rewriter.create<IREE::Stream::TensorConstantOp>(
            globalOp.getLoc(), resourceOp.getType(), initialValueAttr,
            TypeAttr::get(globalOp.getType()),
            /*result_encoding_dims=*/ValueRange{}, affinityAttr);
        initialValueSize = rewriter.create<IREE::Stream::ResourceSizeOp>(
            globalOp.getLoc(), indexType, initialValue);
      }
      resourceOp.createStoreOp(globalOp.getLoc(), initialValue, rewriter);
      resourceSizeOp.createStoreOp(globalOp.getLoc(), initialValueSize,
                                   rewriter);
      rewriter.create<IREE::Util::ReturnOp>(globalOp.getLoc());
    }

    expansionState->globalMap[globalOp.getSymName()] = ExpandedGlobalResource{
        resourceOp,
        resourceSizeOp,
    };

    return success();
  }
};

struct GlobalLoadOpExpansion
    : public BaseGlobalConversionPattern<IREE::Util::GlobalLoadOp> {
  using BaseGlobalConversionPattern::BaseGlobalConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only apply to expanded types (tensors/etc).
    if (!isExpandedType(loadOp.getType()))
      return failure();
    auto &expandedGlobal = this->expansionState->globalMap[adaptor.getGlobal()];

    // Insert a load/transfer to the unknown resource lifetime.
    auto unknownType = IREE::Stream::ResourceType::get(rewriter.getContext());
    auto resource =
        rewriter
            .create<IREE::Util::GlobalLoadOp>(
                loadOp.getLoc(), expandedGlobal.resourceOp.getType(),
                expandedGlobal.resourceOp.getSymName())
            .getResult();
    auto resourceSize = rewriter
                            .create<IREE::Util::GlobalLoadOp>(
                                loadOp.getLoc(), rewriter.getIndexType(),
                                expandedGlobal.resourceSizeOp.getSymName())
                            .getResult();
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        loadOp, unknownType, resource, resourceSize, resourceSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);

    return success();
  }
};

struct GlobalStoreOpExpansion
    : public BaseGlobalConversionPattern<IREE::Util::GlobalStoreOp> {
  using BaseGlobalConversionPattern::BaseGlobalConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::GlobalStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only apply to expanded types (tensors/etc).
    if (!isExpandedType(storeOp.getValue().getType()))
      return failure();
    auto &expandedGlobal = expansionState->globalMap[adaptor.getGlobal()];

    // Insert a transfer/store to the global with unknown lifetime. Lifetime
    // refinement will make this go away if possible.
    auto value =
        consumeTensorOperand(storeOp.getLoc(), adaptor.getValue(), rewriter);
    assert(expandedGlobal.resourceOp && "Missing resource op");
    auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
        storeOp.getLoc(), expandedGlobal.resourceOp.getType(), value.resource,
        value.resourceSize, value.resourceSize, /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
    rewriter.replaceOpWithNewOp<IREE::Util::GlobalStoreOp>(
        storeOp, transferOp.getResult(),
        expandedGlobal.resourceOp.getSymName());
    rewriter.create<IREE::Util::GlobalStoreOp>(
        storeOp.getLoc(), value.resourceSize,
        expandedGlobal.resourceSizeOp.getSymName());

    return success();
  }
};

} // namespace

void populateUtilToStreamConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns
      .insert<FuncOpSignatureConversion, CallOpConversion, ReturnOpConversion>(
          typeConverter, context);

  auto expansionState = std::make_shared<GlobalExpansionState>();
  // TODO(#7432): add indirect global expansion support to streams.
  patterns
      .insert<GlobalOpExpansion, GlobalLoadOpExpansion, GlobalStoreOpExpansion>(
          expansionState, typeConverter, context);
  patterns.add<GenericConvertTypesPattern<IREE::Util::GlobalOp>,
               GenericConvertTypesPattern<IREE::Util::GlobalLoadOp>,
               GenericConvertTypesPattern<IREE::Util::GlobalStoreOp>>(
      typeConverter, context);
}

void populateUtilToStreamConversionPatterns(MLIRContext *context,
                                            ConversionTarget &conversionTarget,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  typeConverter.addConversion([=](IREE::Util::PtrType type,
                                  SmallVectorImpl<Type> &resultTypes) {
    // Expand pointers to tensors to [resource, sizeof resource] pointers.
    if (!isExpandedType(type))
      return failure();
    resultTypes.push_back(
        IREE::Util::PtrType::get(IREE::Stream::ResourceType::get(context)));
    resultTypes.push_back(IREE::Util::PtrType::get(IndexType::get(context)));
    return success();
  });

  typeConverter.addConversion(
      [=](IREE::Util::PtrType type, SmallVectorImpl<Type> &resultTypes) {
        // Expand pointers to tensors to [ptr<resource>, ptr<sizeof resource>].
        if (!isExpandedType(type.getTargetType()))
          return failure();
        resultTypes.push_back(IREE::Stream::ResourceType::get(context));
        resultTypes.push_back(IndexType::get(context));
        return success();
      });

  conversionTarget.addLegalOp<IREE::Util::InitializerOp>();
  conversionTarget.addDynamicallyLegalOp<IREE::Util::FuncOp>(
      [&](IREE::Util::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
  addGenericLegalOp<IREE::Util::CallOp>(conversionTarget, typeConverter);
  addGenericLegalOp<IREE::Util::ReturnOp>(conversionTarget, typeConverter);

  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalOp>(
      [&](IREE::Util::GlobalOp op) {
        return typeConverter.isLegal(op.getType()) &&
               (!op.getInitialValueAttr() ||
                !llvm::isa<TensorType>(op.getInitialValueAttr().getType()));
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalAddressOp>(
      [&](IREE::Util::GlobalAddressOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalLoadOp>(
      [&](IREE::Util::GlobalLoadOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalLoadIndirectOp>(
      [&](IREE::Util::GlobalLoadIndirectOp op) {
        return typeConverter.isLegal(op.getResult().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalStoreOp>(
      [&](IREE::Util::GlobalStoreOp op) {
        return typeConverter.isLegal(op.getValue().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::GlobalStoreIndirectOp>(
      [&](IREE::Util::GlobalStoreIndirectOp op) {
        return typeConverter.isLegal(op.getValue().getType());
      });
  conversionTarget.addDynamicallyLegalOp<IREE::Util::OptimizationBarrierOp>(
      [&](IREE::Util::OptimizationBarrierOp op) {
        return typeConverter.isLegal(op.getResultTypes());
      });

  populateUtilToStreamConversionPatterns(context, typeConverter, patterns);
}

} // namespace mlir::iree_compiler
