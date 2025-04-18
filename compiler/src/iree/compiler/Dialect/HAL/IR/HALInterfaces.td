// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_INTERFACES
#define IREE_DIALECT_HAL_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// IREE::HAL::DeviceInitializationAttrInterface
//===----------------------------------------------------------------------===//

def HAL_DeviceInitializationAttrInterface :
    AttrInterface<"DeviceInitializationAttrInterface", [
      TypedAttrInterface,
    ]> {
  let description = [{
    Interface for attributes controlling device initialization.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        prints a string description of the initialization specification for
        inclusion in error messages. May include internal newlines but no
        newline is expected at the end.
      }],
      /*retTy=*/"void",
      /*methodName=*/"printStatusDescription",
      /*args=*/(ins "llvm::raw_ostream &":$os),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Builds a `util.initializer` body responsible for initializing a device
        global. Returns the device value that should be stored into the global.
        The name provided is an informal identifier that can be used to produce
        user-level error messages that reference the device.

        The provided `buildDeviceTargetMatch` function will be called with a
        `!hal.device` SSA value and a device target specification and should
        return an `i1` value indicating whether the given device matches the
        specification. If the device always matches (rare!) a null value may
        be returned.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"buildDeviceEnumeration",
      /*args=*/(ins
        "Location":$loc,
        "IREE::HAL::BuildDeviceTargetMatchFn":$buildDeviceTargetMatch,
        "OpBuilder &":$builder
      ),
      /*methodBody=*/[{}]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::HAL::DeviceQueueOpInterface
//===----------------------------------------------------------------------===//

def HAL_DeviceQueueOp : OpInterface<"DeviceQueueOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";

  let description = [{
    Interface for queued device operations.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the fence that must be waited on before the op executes on the
        queue, if any.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getQueueWaitFence",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getWaitFence();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the fence that must be waited on before the op executes on the
        queue, if any.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setQueueWaitFence",
      /*args=*/(ins "Value":$value),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        $_op.getWaitFenceMutable().assign(value);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the fence that will be signaled after the op executes on the
        queue, if any.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getQueueSignalFence",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getSignalFence();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the fence that will be signaled after the op executes on the
        queue, if any.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setQueueSignalFence",
      /*args=*/(ins "Value":$value),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        $_op.getSignalFenceMutable().assign(value);
      }]
    >,
  ];
}

#endif  // IREE_DIALECT_HAL_INTERFACES
