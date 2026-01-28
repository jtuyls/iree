# Hoistable Dispatch Infrastructure

This document summarizes the changes made to implement a new hoistable dispatch infrastructure in the IREE compiler. The goal is to wrap encoding operations (`iree_encoding.set_encoding` and `iree_encoding.unset_encoding`) in dedicated hoistable dispatch ops, providing cleaner separation of concerns and enabling better optimization opportunities.

## Overview

### Before These Changes

Previously, encoding operations were either:
1. Placed inside `flow.dispatch.workgroups` along with other compute operations
2. Converted directly to `stream.tensor.encode` ops at the Stream level

### After These Changes

Now, encoding operations follow a dedicated path:
1. Wrapped in `flow.hoistable_dispatch` at the Flow level
2. Converted to `stream.hoistable_dispatch` during Flow→Stream conversion
3. Unified across globals by `UnifyEncodingForGlobals` pass
4. Specialized by `SpecializeEncodings` pass
5. Materialized into `stream.async.dispatch` by `MaterializeEncodings` pass

## New Operations

### `flow.hoistable_dispatch`

A Flow dialect operation that encapsulates pure operations with implicit capture. The key property is that it can be hoisted into initializers for globals:

```mlir
%result = flow.hoistable_dispatch(%input : tensor<4x5xf32>) -> (tensor<4x5xf32, #encoding>) {
  %encoded = iree_encoding.set_encoding %input : tensor<4x5xf32> -> tensor<4x5xf32, #encoding>
  flow.return %encoded : tensor<4x5xf32, #encoding>
}
```

### `stream.hoistable_dispatch`

The Stream dialect counterpart that operates on stream resources:

```mlir
%result = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
  (%input : tensor<4x5xf32> in !stream.resource<*>{%input_size})
  -> (tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}) {
    %enc = stream.tensor.encode on(#hal.device.affinity<@device>)
      %input : tensor<4x5xf32> in !stream.resource<*>{%input_size}
      -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}
    stream.yield %enc : !stream.resource<*>{%result_size}
}
```

## IR Transformation Examples

### 1. After `FuseEncodingOpsIntoDispatchRegions` Pass

**Input IR:**
```mlir
util.func public @encode_tensor(%arg0: tensor<4x5xf32>) -> tensor<4x5xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<4x5xf32> -> tensor<4x5xf32, #encoding>
  util.return %0 : tensor<4x5xf32, #encoding>
}
```

**Output IR:**
```mlir
util.func public @encode_tensor(%arg0: tensor<4x5xf32>) -> tensor<4x5xf32, #encoding> {
  %0 = flow.hoistable_dispatch(%arg0 : tensor<4x5xf32>) -> (tensor<4x5xf32, #encoding>) {
    %1 = iree_encoding.set_encoding %arg0 : tensor<4x5xf32> -> tensor<4x5xf32, #encoding>
    flow.return %1 : tensor<4x5xf32, #encoding>
  }
  util.return %0 : tensor<4x5xf32, #encoding>
}
```

### 2. After `FlowToStream` Conversion

**Input IR (Flow level):**
```mlir
%0 = flow.hoistable_dispatch(%arg0 : tensor<4x5xf32>) -> (tensor<4x5xf32, #encoding>) {
  %1 = iree_encoding.set_encoding %arg0 : tensor<4x5xf32> -> tensor<4x5xf32, #encoding>
  flow.return %1 : tensor<4x5xf32, #encoding>
}
```

**Output IR (Stream level):**
```mlir
%input_size = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<4x5xf32> : index
%result_size = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<4x5xf32, #encoding> : index
%0 = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
  (%arg0 : tensor<4x5xf32> in !stream.resource<*>{%input_size})
  -> (tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}) {
    %enc = stream.tensor.encode on(#hal.device.affinity<@device>)
      %arg0 : tensor<4x5xf32> in !stream.resource<*>{%input_size}
      -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}
    stream.yield %enc : !stream.resource<*>{%result_size}
}
```

### 3. After `UnifyEncodingForGlobals` Pass

This pass unifies multiple encodings of the same immutable global into a single encoding.

**Input IR:**
```mlir
util.global private @weights : !stream.resource<constant>

util.initializer {
  %weights = util.global.load @weights : !stream.resource<constant>
  %size = stream.resource.size %weights : !stream.resource<constant>
  
  // First encoding with resolver A
  %enc1 = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
    (%weights : tensor<4096x4096xf32> in !stream.resource<constant>{%size})
    -> (tensor<4096x4096xf32, #encoding_a> in !stream.resource<*>{%dest_size_a}) {
      %e = stream.tensor.encode ...
      stream.yield %e : !stream.resource<*>{%dest_size_a}
  }
  
  // Second encoding with resolver B  
  %enc2 = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
    (%weights : tensor<4096x4096xf32> in !stream.resource<constant>{%size})
    -> (tensor<4096x4096xf32, #encoding_b> in !stream.resource<*>{%dest_size_b}) {
      %e = stream.tensor.encode ...
      stream.yield %e : !stream.resource<*>{%dest_size_b}
  }
  ...
}
```

**Output IR:**
```mlir
util.global private @weights : !stream.resource<constant>

util.initializer {
  %weights = util.global.load @weights : !stream.resource<constant>
  %size = stream.resource.size %weights : !stream.resource<constant>
  
  // Unified encoding with identity resolver (to be specialized later)
  %enc = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
    (%weights : tensor<4096x4096xf32> in !stream.resource<constant>{%size})
    -> (tensor<4096x4096xf32, #encoding_identity> in !stream.resource<*>{%dest_size}) {
      %e = stream.tensor.encode ...
      stream.yield %e : !stream.resource<*>{%dest_size}
  }
  // Both uses now reference the same unified encoding
  ...
}
```

### 4. After `MaterializeEncodings` Pass

This pass converts `stream.hoistable_dispatch` ops into `stream.async.dispatch` ops with executable definitions.

**Input IR:**
```mlir
%0 = stream.hoistable_dispatch on(#hal.device.affinity<@device>)
  (%resource : tensor<4x5xf32> in !stream.resource<*>{%input_size})
  -> (tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}) {
    %enc = stream.tensor.encode on(#hal.device.affinity<@device>)
      %resource : tensor<4x5xf32> in !stream.resource<*>{%input_size}
      -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%result_size}
    stream.yield %enc : !stream.resource<*>{%result_size}
}
```

**Output IR:**
```mlir
stream.executable private @_encoding_0 {
  stream.executable.export public @_encoding_0_encode_4x5xf32_to_4x5xf32 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_encoding_0_encode_4x5xf32_to_4x5xf32(%src: !stream.binding, %dest: !stream.binding) {
      %c0 = arith.constant 0 : index
      %src_tensor = stream.binding.subspan %src[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5xf32>>
      %dest_tensor = stream.binding.subspan %dest[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5xf32, #encoding>>
      %val = iree_tensor_ext.dispatch.tensor.load %src_tensor, offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5xf32>> -> tensor<4x5xf32>
      %encoded = iree_encoding.set_encoding %val : tensor<4x5xf32> -> tensor<4x5xf32, #encoding>
      iree_tensor_ext.dispatch.tensor.store %encoded, %dest_tensor, offsets = [0, 0], sizes = [4, 5], strides = [1, 1]
        : tensor<4x5xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5xf32, #encoding>>
      return
    }
  }
}

%0 = stream.async.dispatch on(#hal.device.affinity<@device>) 
  @_encoding_0::@_encoding_0_encode_4x5xf32_to_4x5xf32(%resource[%c0 to %input_size for %input_size]) 
  : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%result_size}
```

## Files Modified

### New/Modified Operations

| File | Changes |
|------|---------|
| `Flow/IR/FlowOps.td` | Defined `flow.hoistable_dispatch` op |
| `Flow/IR/FlowOps.cpp` | Implemented verifier, parser, printer for the op |
| `Stream/IR/StreamOps.td` | Defined `stream.hoistable_dispatch` op, updated `stream.yield` |
| `Stream/IR/StreamOps.cpp` | Implemented verifier for the op |

### Passes Modified

| Pass | File | Changes |
|------|------|---------|
| `FuseEncodingOpsIntoDispatchRegions` | `DispatchCreation/FuseEncodingOpsIntoDispatchRegions.cpp` | Creates `flow.hoistable_dispatch` ops |
| `FlowToStream` | `Stream/Conversion/FlowToStream/Patterns.cpp` | Converts `flow.hoistable_dispatch` to `stream.hoistable_dispatch` |
| `UnifyEncodingForGlobals` | `Stream/Transforms/UnifyEncodingForGlobals.cpp` | Updated to work with nested encoding ops |
| `MaterializeEncodings` | `Stream/Transforms/MaterializeEncodings.cpp` | Converts `stream.hoistable_dispatch` to `stream.async.dispatch` |

### Tests Updated

| Test File | Changes |
|-----------|---------|
| `DispatchCreation/test/fuse_encoding_ops_into_dispatch_regions.mlir` | Updated for new op structure |
| `DispatchCreation/test/pipeline_tests.mlir` | Updated CHECK lines |
| `DispatchCreation/test/set_encoding_pipeline.mlir` | Updated to expect `flow.hoistable_dispatch` |
| `Stream/Transforms/test/unify_encoding_for_globals.mlir` | Updated all 14 test cases for new structure |
| `Stream/Transforms/test/materialize_encodings.mlir` | Recreated for new architecture |

## End-to-End Example: Matmul with Constant Weights

This example demonstrates how constant weights are hoisted to the initializer while dynamic inputs are encoded at runtime.

**Input IR:**
```mlir
util.func public @matmul_static(%arg0: tensor<64x128xf32>) -> tensor<64x256xf32> {
  %cst = arith.constant dense<1.0> : tensor<128x256xf32>
  %cst_0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x256xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x256xf32>) -> tensor<64x256xf32>
  %2 = linalg.matmul ins(%arg0, %cst : tensor<64x128xf32>, tensor<128x256xf32>) 
                     outs(%1 : tensor<64x256xf32>) -> tensor<64x256xf32>
  util.return %2 : tensor<64x256xf32>
}
```

**After compilation with `--compile-to=flow`:**
```mlir
// Constant weights are hoisted to a global and encoded in the initializer
util.global private @__hoisted_tensor_128x256xf32__encoded : tensor<128x256xf32, #encoding_rhs>

util.initializer {
  %cst = arith.constant dense<1.0> : tensor<128x256xf32>
  // Hoistable dispatch for constant - runs once at initialization
  %0 = flow.hoistable_dispatch(%cst : tensor<128x256xf32>) -> (tensor<128x256xf32, #encoding_rhs>) {
    %1 = iree_encoding.set_encoding %cst : tensor<128x256xf32> -> tensor<128x256xf32, #encoding_rhs>
    flow.return %1 : tensor<128x256xf32, #encoding_rhs>
  }
  util.global.store %0, @__hoisted_tensor_128x256xf32__encoded : tensor<128x256xf32, #encoding_rhs>
  util.return
}

util.func public @matmul_static(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // Load pre-encoded constant weights (no runtime encoding needed!)
  %weights = util.global.load immutable @__hoisted_tensor_128x256xf32__encoded : tensor<128x256xf32, #encoding_rhs>
  
  %input = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<64x128xf32>
  
  // Only ONE hoistable dispatch at runtime - for the dynamic input
  %encoded_input = flow.hoistable_dispatch(%input : tensor<64x128xf32>) -> (tensor<64x128xf32, #encoding_lhs>) {
    %enc = iree_encoding.set_encoding %input : tensor<64x128xf32> -> tensor<64x128xf32, #encoding_lhs>
    flow.return %enc : tensor<64x128xf32, #encoding_lhs>
  }
  
  // Matmul dispatch uses pre-encoded weights
  %result = flow.dispatch @matmul_dispatch(%encoded_input, %weights) : 
    (tensor<64x128xf32, #encoding_lhs>, tensor<128x256xf32, #encoding_rhs>) -> tensor<64x256xf32>
  
  %output = hal.tensor.export %result "output0" : tensor<64x256xf32> -> !hal.buffer_view
  util.return %output : !hal.buffer_view
}
```

**Key observations:**
- The constant tensor encoding is hoisted to `util.initializer` - runs once at module load
- Only ONE `flow.hoistable_dispatch` appears in the main function - for the dynamic input
- The matmul dispatch receives pre-encoded weights from the global

### After Stream Conversion and MaterializeEncodings (`--compile-to=stream`)

Both `flow.hoistable_dispatch` ops (the one hoisted to the initializer for constants, and the one remaining in the main function for dynamic inputs) go through the same transformation:

1. **FlowToStream**: `flow.hoistable_dispatch` → `stream.hoistable_dispatch`
2. **SpecializeEncodings**: Abstract encodings → Concrete GPU-specific layouts
3. **MaterializeEncodings**: `stream.hoistable_dispatch` → `stream.executable` + `stream.cmd.dispatch`

Each `stream.hoistable_dispatch` creates its own `stream.executable` with encoding logic. The only difference is **when** they execute:
- **Constant weights** (`@_encoding_0`): Once at module initialization, result stored in global
- **Dynamic input** (`@_encoding_1`): Every time the function is called

**Encoding Executable for constant weights (128x256, from initializer):**
```mlir
stream.executable private @_encoding_0 {
  stream.executable.export public @_encoding_0_encode_128x256xf32_to_128x256xf32 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_encoding_0_encode_128x256xf32_to_128x256xf32(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x256xf32, #iree_encoding.layout<[...]>>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] 
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
      // Encoding is now specialized with GPU-specific layout (tiling, swizzle, etc.)
      %3 = iree_encoding.set_encoding %2 : tensor<128x256xf32> 
        -> tensor<128x256xf32, #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<
             configuration = {encoding_info = {
               innerDimsPos = [1, 0], innerTileSizes = [16, 16], 
               outerDimsPerm = [1, 0], swizzle = {...}
             }}>]>>
      iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : ...
      return
    }
  }
}
```

**Encoding Executable for dynamic input (64x128, from main function):**
```mlir
stream.executable private @_encoding_1 {
  stream.executable.export public @_encoding_1_encode_64x128xf32_to_64x128xf32 workgroups() -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_encoding_1_encode_64x128xf32_to_64x128xf32(%arg0: !stream.binding, %arg1: !stream.binding) {
      // Same pattern: load unencoded tensor, apply encoding, store encoded result
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding 
        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32, #iree_encoding.layout<[...]>>>
      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] 
        : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
      %3 = iree_encoding.set_encoding %2 : tensor<64x128xf32> 
        -> tensor<64x128xf32, #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<...>]>>
      iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : ...
      return
    }
  }
}
```

**Initializer becomes a `stream.cmd.execute` block:**
```mlir
util.global private @__hoisted_tensor_128x256xf32__encoded : !stream.resource<constant>

util.initializer {
  %c131072 = arith.constant 131072 : index  // encoded size
  %c1065353216_i32 = arith.constant 1065353216 : i32  // 1.0f in IEEE 754
  
  // Allocate output (constant) and temp (transient) buffers
  %result, %result_timepoint = stream.resource.alloca uninitialized ... : !stream.resource<constant>{%c131072}
  %temp, %temp_timepoint = stream.resource.alloca uninitialized ... : !stream.resource<transient>{%c131072}
  
  %1 = stream.cmd.execute once on(#hal.device.affinity<@__device_0>) await(%0) 
    => with(%result as %arg0: !stream.resource<constant>{%c131072}, 
            %temp as %arg1: !stream.resource<transient>{%c131072}) {
    // Fill temp buffer with constant 1.0f
    stream.cmd.fill %c1065353216_i32, %arg1[%c0 for %c131072] : i32 -> !stream.resource<transient>{%c131072}
    
    // Dispatch encoding kernel to encode weights
    stream.cmd.dispatch @_encoding_0::@_encoding_0_encode_128x256xf32_to_128x256xf32 {
      ro %arg1[%c0 for %c131072] : !stream.resource<transient>{%c131072},  // source (unencoded)
      wo %arg0[%c0 for %c131072] : !stream.resource<constant>{%c131072}    // dest (encoded)
    }
  } => !stream.timepoint
  
  // Store encoded result to global - available for all future calls
  %3 = stream.timepoint.await sync %2 => %result : !stream.resource<constant>{%c131072}
  util.global.store %3, @__hoisted_tensor_128x256xf32__encoded : !stream.resource<constant>
  util.return
}
```

**Main function - dynamic input encoding runs every call:**
```mlir
util.func public @matmul_static(%arg0: !hal.buffer_view) -> !hal.buffer_view {
  // Load pre-encoded constant (no runtime encoding needed for weights!)
  %__hoisted = util.global.load immutable @__hoisted_tensor_128x256xf32__encoded : !stream.resource<constant>
  
  %0 = stream.tensor.import ... %arg0 : !hal.buffer_view -> tensor<64x128xf32> in !stream.resource<external>{%c32768}
  
  %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%1) 
    => with(%0 as %arg1: !stream.resource<external>{%c32768}, 
            %__hoisted as %arg2: !stream.resource<constant>{%c131072},
            %result as %arg3: !stream.resource<external>{%c65536},
            %temp as %arg4: !stream.resource<transient>{%c32768}) {
    // This is the materialized hoistable_dispatch for the dynamic input
    // It runs every time the function is called (can't be hoisted - depends on runtime data)
    stream.cmd.dispatch @_encoding_1::@_encoding_1_encode_64x128xf32_to_64x128xf32 {
      ro %arg1[%c0 for %c32768] : !stream.resource<external>{%c32768},
      wo %arg4[%c0 for %c32768] : !stream.resource<transient>{%c32768}
    }
    // Matmul uses freshly-encoded input + pre-encoded weights from global
    stream.cmd.dispatch @matmul_static_dispatch_0::@matmul_static_dispatch_0_matmul_64x256x128_f32 {
      ro %arg4[%c0 for %c32768] : !stream.resource<transient>{%c32768},   // encoded input
      ro %arg2[%c0 for %c131072] : !stream.resource<constant>{%c131072}, // pre-encoded weights
      wo %arg3[%c0 for %c65536] : !stream.resource<external>{%c65536}    // output
    }
  } => !stream.timepoint
  
  %5 = stream.tensor.export ... %4 : tensor<64x256xf32> in !stream.resource<external>{%c65536} -> !hal.buffer_view
  util.return %5 : !hal.buffer_view
}
```

**Summary:**

Both `hoistable_dispatch` ops create their own `stream.executable` and are materialized to `stream.cmd.dispatch` calls. The optimization benefit is that the constant weights are encoded **once at module initialization** and reused, while only the dynamic input requires encoding on each function call.

## Key Design Decisions

1. **Implicit Capture**: `flow.hoistable_dispatch` uses implicit capture (no `IsolatedFromAbove` trait) to simplify the op structure and allow direct references to values defined outside the region.

2. **stream.yield Terminator**: `stream.hoistable_dispatch` uses `stream.yield` as its terminator (not `stream.return`), allowing it to yield values back to the parent op.

3. **stream.async.clone Support**: The verifier for `stream.hoistable_dispatch` explicitly allows `stream.async.clone` ops inside the region, as they may be introduced by the `MaterializeCopyOnWrite` pass before encoding materialization.

4. **Encoding Unification**: The `UnifyEncodingForGlobals` pass now operates on `stream.hoistable_dispatch` ops, looking for nested `stream.tensor.encode` ops to determine the encoding being applied to globals.
