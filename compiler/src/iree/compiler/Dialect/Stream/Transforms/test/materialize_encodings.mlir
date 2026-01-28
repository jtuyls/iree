// RUN: iree-opt --iree-stream-materialize-encodings --split-input-file %s | FileCheck %s

// Test: stream.hoistable_dispatch is materialized to stream.async.dispatch
// with an executable containing the encoding logic.

#encoding = #iree_encoding.testing<>

// CHECK:      stream.executable private @_encoding_0 {
// CHECK:         stream.executable.export public @_encoding_0_encode_4x5xf32_to_4x5xf32 workgroups()
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice()
// CHECK:         func.func @_encoding_0_encode_4x5xf32_to_4x5xf32(
// CHECK-SAME:      %[[SRC_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK-SAME:      %[[DEST_ARG:[a-zA-Z0-9]+]]: !stream.binding
// CHECK:           %[[SRC_BUF:.+]] = stream.binding.subspan %[[SRC_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5xf32>>
// CHECK:           %[[DEST_BUF:.+]] = stream.binding.subspan %[[DEST_ARG]]{{.+}} : !stream.binding
// CHECK-SAME:        -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5xf32, #{{.+}}>>
// CHECK:           %[[VAL:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BUF]]
// CHECK:           %[[ENCODED_VAL:.+]] = iree_encoding.set_encoding %[[VAL]]
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[ENCODED_VAL]], %[[DEST_BUF]]

// CHECK-LABEL: util.func public @encode_static_shape
// CHECK-SAME:    %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[TOTAL_SIZE:[a-zA-Z0-9]+]]
util.func public @encode_static_shape(%resource: !stream.resource<*>, %total_size: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @_encoding_0::@_encoding_0_encode_4x5xf32_to_4x5xf32
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}]) : (!stream.resource<*>{%[[TOTAL_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[TOTAL_SIZE]]}
  %0 = stream.hoistable_dispatch on(#hal.device.affinity<@device_a>)
    (%resource : tensor<4x5xf32> in !stream.resource<*>{%total_size})
    -> (tensor<4x5xf32, #encoding> in !stream.resource<*>{%total_size}) {
      %enc = stream.tensor.encode on(#hal.device.affinity<@device_a>)
        %resource : tensor<4x5xf32> in !stream.resource<*>{%total_size}
        -> tensor<4x5xf32, #encoding> in !stream.resource<*>{%total_size}
      stream.yield %enc : !stream.resource<*>{%total_size}
  }
  util.return %0 : !stream.resource<*>
}

// -----

// Test: encoding dispatch region with dynamic dimensions.

#encoding = #iree_encoding.testing<>

// CHECK:      stream.executable private @_encoding_0 {
// CHECK:         stream.executable.export public @_encoding_0_encode_DxDxf32_to_DxDxf32 workgroups(
// CHECK-SAME:      %{{.+}}: index, %{{.+}}: index, %{{.+}}: index, %{{.+}}: index)
// CHECK-NEXT:      iree_tensor_ext.dispatch.workgroup_count_from_slice
// CHECK:         func.func @_encoding_0_encode_DxDxf32_to_DxDxf32(
// CHECK-SAME:      %{{.+}}: !stream.binding

// CHECK-LABEL: util.func public @encode_dynamic_shape
// CHECK-SAME:    %[[RESOURCE:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[SRC_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST_SIZE:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[D1:[a-zA-Z0-9]+]]
util.func public @encode_dynamic_shape(%resource: !stream.resource<*>, %src_size: index, %dest_size: index, %d0: index, %d1: index) -> !stream.resource<*> {
  // CHECK:      stream.async.dispatch on(#{{.+}}) @_encoding_0::@_encoding_0_encode_DxDxf32_to_DxDxf32[%[[D0]], %[[D1]], %[[D0]], %[[D1]]]
  // CHECK-SAME:   (%[[RESOURCE]][{{.+}}], %[[D0]], %[[D1]], %[[D0]], %[[D1]]) : (!stream.resource<*>{%[[SRC_SIZE]]}
  // CHECK-SAME:   -> !stream.resource<*>{%[[DEST_SIZE]]}
  %0 = stream.hoistable_dispatch on(#hal.device.affinity<@device_a>)
    (%resource : tensor<?x?xf32>{%d0, %d1} in !stream.resource<*>{%src_size})
    -> (tensor<?x?xf32, #encoding>{%d0, %d1} in !stream.resource<*>{%dest_size}) {
      %enc = stream.tensor.encode on(#hal.device.affinity<@device_a>)
        %resource : tensor<?x?xf32>{%d0, %d1} in !stream.resource<*>{%src_size}
        -> tensor<?x?xf32, #encoding>{%d0, %d1} in !stream.resource<*>{%dest_size}
      stream.yield %enc : !stream.resource<*>{%dest_size}
  }
  util.return %0 : !stream.resource<*>
}

// -----

// Test: encoding dispatch region inside initializer (for hoisted encoding ops).

#encoding = #iree_encoding.testing<>

util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
util.global private @encoded : !stream.resource<constant>

// CHECK:      stream.executable private @_encoding_0 {
// CHECK:         stream.executable.export public @_encoding_0_encode_4096x4096xf32_to_4096x4096xf32

// CHECK: util.initializer
util.initializer {
  %source = util.global.load @source : !stream.resource<constant>
  %source_size = stream.resource.size %source : !stream.resource<constant>
  %dest_size = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding> : index

  // CHECK: stream.async.dispatch on(#{{.+}}) @_encoding_0::@_encoding_0_encode_4096x4096xf32_to_4096x4096xf32
  %enc = stream.hoistable_dispatch on(#hal.device.affinity<@device_a>)
    (%source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size})
    -> (tensor<4096x4096xf32, #encoding> in !stream.resource<*>{%dest_size}) {
      %inner = stream.tensor.encode on(#hal.device.affinity<@device_a>)
        %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size}
        -> tensor<4096x4096xf32, #encoding> in !stream.resource<*>{%dest_size}
      stream.yield %inner : !stream.resource<*>{%dest_size}
  }

  %const = stream.async.clone on(#hal.device.affinity<@device_a>) %enc : !stream.resource<*>{%dest_size} -> !stream.resource<constant>{%dest_size}
  util.global.store %const, @encoded : !stream.resource<constant>
  util.return
}
