// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_OPS
#define IREE_DIALECT_ENCODING_OPS

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "iree/compiler/Dialect/Encoding/IR/EncodingInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// set_encoding op.
//===----------------------------------------------------------------------===//

def IREEEncoding_SetEncodingOp : IREEEncoding_PureOp<"set_encoding",[
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
    DeclareOpInterfaceMethods<IREEEncoding_EncodingDimReificationInterface,
      ["reifyEncodingDim"]>,
    Pure
  ]> {
  let summary = [{Perform pack and pad operation on source.}];
  let description = [{
    Operation to assign an encoding to a tensor. The operation does not change
    the rank or extent of a tensor. Instead it adds a LayoutResolverAttr
    attribute to the tensor type to represent a change in layout.

    The optional `encoding_dims` operand carries dynamic values needed by the
    encoding (e.g., M, N, K dimensions for matmul encodings). These values are
    used for runtime layout selection based on problem size.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$encoding_dims);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $source (`encoding_dims` `{` $encoding_dims^ `}`)? `:` type($source) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return cast<RankedTensorType>(getSource().getType());
    }
    RankedTensorType getResultType() {
      return cast<RankedTensorType>(getResult().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// unset_encoding op.
//===----------------------------------------------------------------------===//

def IREEEncoding_UnsetEncodingOp : IREEEncoding_PureOp<"unset_encoding", [
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface, ["reifyResultShapes"]>,
    AttrSizedOperandSegments, Pure
  ]> {
  let summary = [{Perform unpack and extract operation on source.}];
  let description = [{
    Operation to convert a tensor with LayoutResolverAttr encoding that
    represents its data layout into a tensor with default layout
    (i.e. no encoding). For now in IREE the default layout is row-major.

    The optional `encoding_dims` operand carries dynamic values needed by the
    encoding (e.g., M, N, K dimensions for matmul encodings). These values are
    used for runtime layout selection based on problem size.
  }];
  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$result_dims,
    Variadic<Index>:$encoding_dims);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict $source (`encoding_dims` `{` $encoding_dims^ `}`)? `:` type($source) `->` type($result) (`` `{` $result_dims^ `}`)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return cast<RankedTensorType>(getSource().getType());
    }
    RankedTensorType getResultType() {
      return cast<RankedTensorType>(getResult().getType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// dim op.
//===----------------------------------------------------------------------===//

def IREEEncoding_DimOp : IREEEncoding_PureOp<"dim", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
  ]> {
  let summary = [{Query encoding dimension from an encoded tensor.}];
  let description = [{
    Returns the value of an encoding-specific dimension from a tensor with
    encoding. Unlike `tensor.dim` which queries shape dimensions, this operation
    queries the semantic dimensions captured in the encoding.

    For matmul encodings, the encoding dimensions are M, N, K (in that order):
    - index 0 returns M
    - index 1 returns N
    - index 2 returns K

    For a matmul LHS tensor with shape `[M, K]`, `tensor.dim` can only query
    M and K, but `iree_encoding.dim` can query M, N, and K since all three are
    captured in the encoding.

    This operation is resolved by tracing the producer chain:
    - From `set_encoding`: returns the corresponding `encoding_dims` value
    - Through `tensor.cast` / DPS ops: forwards query to source

    Example:

    ```mlir
    // Query the M dimension (index 0) of a matmul LHS encoding
    %m = iree_encoding.dim %encoded[0] : tensor<?x?xf32, #encoding>

    // Query the N dimension (index 1) - available even though not in LHS shape
    %n = iree_encoding.dim %encoded[1] : tensor<?x?xf32, #encoding>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    IndexAttr:$index
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict $source `[` $index `]` `:` type($source)
  }];

  let builders = [
    OpBuilder<(ins "Value":$source, "int64_t":$index)>
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer.
    int64_t getConstantIndex() { return getIndex().getSExtValue(); }
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_ENCODING_OPS
