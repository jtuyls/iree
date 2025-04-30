// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/ExternalInterfaces/CodegenExternalModels.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"

namespace mlir::iree_compiler::IREE::Codegen {

struct EncodingNopHostEncodingLayoutResolverAttrInterface final
    : IREE::Encoding::EncodingLayoutResolverAttrInterface::ExternalModel<
          EncodingNopHostEncodingLayoutResolverAttrInterface,
          EncodingNopLayoutAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr,
                                      DictionaryAttr config) const {
    return EncodingNopLayoutAttr::get(attr.getContext());
  }

  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    return EncodingNopLayoutAttr::get(attr.getContext());
  }
};

struct EncodingNopHostSerializableEncodingAttrInterface final
    : IREE::Encoding::SerializableEncodingAttrInterface::ExternalModel<
          EncodingNopHostSerializableEncodingAttrInterface,
          EncodingNopLayoutAttr> {
public:
  Type convertType(Attribute attr, Type type) const {
    EncodingNopLayoutAttr layoutAttr = cast<EncodingNopLayoutAttr>(attr);
    return TypeSwitch<Type, Type>(type)
        .Case<RankedTensorType>([&](auto rankedTensorType) {
          return rankedTensorType.dropEncoding();
        })
        .Case<DispatchTensorType>([&](auto dispatchTensorType) {
          auto boundType =
              dyn_cast<RankedTensorType>(dispatchTensorType.getBoundType());
          if (!boundType || !boundType.getEncoding()) {
            return dispatchTensorType;
          }
          Type convertedBoundType = convertType(attr, boundType);
          return DispatchTensorType::get(dispatchTensorType.getAccess(),
                                         convertedBoundType);
        })
        .Default([&](auto concreteType) { return concreteType; });
  }
};

void registerCodegenExternalModels(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::Codegen::IREECodegenDialect *dialect) {
        EncodingNopLayoutAttr::attachInterface<
            EncodingNopHostEncodingLayoutResolverAttrInterface,
            EncodingNopHostSerializableEncodingAttrInterface>(*ctx);
      });
}

} // namespace mlir::iree_compiler::IREE::Codegen
