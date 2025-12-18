// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-insert-encoding-specialization-markers))' %s \
// RUN:   | FileCheck %s

// Tests for InsertEncodingSpecializationMarkers pass.
// This pass inserts util.specialize markers for encodings that implement
// DynamicLayoutSpecializerAttr.

// -----

// Test: Non-specializable encoding - no markers inserted

#non_specializable_encoding = #iree_encoding.testing<>

util.func public @non_specializable_encoding(%arg0: tensor<16x16xf32, #non_specializable_encoding>) -> tensor<16x16xf32, #non_specializable_encoding> {
  util.return %arg0 : tensor<16x16xf32, #non_specializable_encoding>
}

// CHECK-LABEL: util.func public @non_specializable_encoding
// CHECK-NOT:     iree_encoding.encoding_dim
// CHECK-NOT:     util.specialize

// -----

// Test: Specializable encoding outside dispatch region - no markers inserted
// Markers only make sense inside dispatch regions where they become workload ordinals

#dynamic_enc = #iree_encoding.dynamic_layout_test<
  1,  // num_dims (single dimension)
  [[#util<int.assumption.array[<umin = 1, umax = 256>]>, 1],
   [#util<int.assumption.array[<umin = 256>]>, 2]],
  0   // fallback seed
>

util.func public @specializable_encoding_no_dispatch(%arg0: tensor<?xf32, #dynamic_enc>) -> tensor<?xf32, #dynamic_enc> {
  util.return %arg0 : tensor<?xf32, #dynamic_enc>
}

// CHECK-LABEL: util.func public @specializable_encoding_no_dispatch
// CHECK-NOT:     iree_encoding.encoding_dim
// CHECK-NOT:     util.specialize

// -----

// Test: Specializable encoding inside flow.dispatch.region

#dynamic_enc_region = #iree_encoding.dynamic_layout_test<
  1,
  [[#util<int.assumption.array[<umin = 1, umax = 128>]>, 1],
   [#util<int.assumption.array[<umin = 128>]>, 2]],
  0
>

util.func public @specializable_in_dispatch_region(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = flow.dispatch.region -> (tensor<?xf32>{%d0}) {
    %encoded = iree_encoding.set_encoding %arg0 : tensor<?xf32> -> tensor<?xf32, #dynamic_enc_region>
    %unencoded = iree_encoding.unset_encoding %encoded : tensor<?xf32, #dynamic_enc_region> -> tensor<?xf32>{%d0}
    flow.return %unencoded : tensor<?xf32>
  }
  util.return %0 : tensor<?xf32>
}

// CHECK-LABEL: util.func public @specializable_in_dispatch_region
// CHECK:         flow.dispatch.region
// CHECK:           %[[ENCODED:.+]] = iree_encoding.set_encoding
// CHECK:           %[[DIM0:.+]] = iree_encoding.encoding_dim %[[ENCODED]][0]
// CHECK:           util.specialize %[[DIM0]] : index

// -----

// Test: Empty variants - no markers inserted

#empty_variants_enc = #iree_encoding.dynamic_layout_test<
  1,  // num_dims
  [], // empty variants
  0   // fallback seed
>

util.func public @empty_variants(%arg0: tensor<?xf32, #empty_variants_enc>) -> tensor<?xf32, #empty_variants_enc> {
  util.return %arg0 : tensor<?xf32, #empty_variants_enc>
}

// CHECK-LABEL: util.func public @empty_variants
// CHECK-NOT:     iree_encoding.encoding_dim
// CHECK-NOT:     util.specialize
