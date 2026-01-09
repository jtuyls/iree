// RUN: iree-opt --split-input-file \
// RUN:   --iree-encoding-enable-dynamic-specialization \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-insert-encoding-specialization-markers))' %s \
// RUN:   | FileCheck %s

// Tests for InsertEncodingSpecializationMarkers pass.
// This pass inserts util.specialize markers for encodings that implement
// DynamicLayoutSpecializerAttr. It creates iree_encoding.dim ops to query
// encoding dimensions from operands with the encoding, wraps them with
// util.specialize markers, and then reifies the dim ops to actual values.

// -----

// Test: Non-specializable encoding - no markers inserted

#non_specializable_encoding = #iree_encoding.testing<>

util.func public @non_specializable_encoding(%arg0: tensor<16x16xf32, #non_specializable_encoding>) -> tensor<16x16xf32, #non_specializable_encoding> {
  util.return %arg0 : tensor<16x16xf32, #non_specializable_encoding>
}

// CHECK-LABEL: util.func public @non_specializable_encoding
// CHECK-NOT:     iree_encoding.dim
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
// CHECK-NOT:     iree_encoding.dim
// CHECK-NOT:     util.specialize

// -----

// Test: Empty variants - no markers inserted since supportsSpecialization returns false

#empty_variants_enc = #iree_encoding.dynamic_layout_test<
  1,  // num_dims
  [], // empty variants
  0   // fallback seed
>

util.func public @empty_variants(%arg0: tensor<?xf32, #empty_variants_enc>) -> tensor<?xf32, #empty_variants_enc> {
  util.return %arg0 : tensor<?xf32, #empty_variants_enc>
}

// CHECK-LABEL: util.func public @empty_variants
// CHECK-NOT:     iree_encoding.dim
// CHECK-NOT:     util.specialize

// -----

// Test: Non-matmul dispatch with specializable encoding
// For element-wise ops, getSpecializationOperands returns the operand with
// the encoding. The dim op is then reified through the operand chain back
// to set_encoding's encoding_dims.

#dynamic_enc_single = #iree_encoding.dynamic_layout_test<
  1,  // num_dims - one dimension to specialize on
  [[#util<int.assumption.array[<umin = 1, umax = 128>]>, 1],
   [#util<int.assumption.array[<umin = 128>]>, 2]],
  0
>

util.func public @non_matmul_dispatch_with_encoding(%arg0: tensor<128xf32>, %dim: index) -> tensor<128xf32, #dynamic_enc_single> {
  // set_encoding outside dispatch with dynamic encoding_dims
  %encoded = iree_encoding.set_encoding %arg0 encoding_dims{%dim} : tensor<128xf32> -> tensor<128xf32, #dynamic_enc_single>
  
  // Dispatch returns encoded tensor - no unset_encoding inside
  %0 = flow.dispatch.region -> (tensor<128xf32, #dynamic_enc_single>) {
    // Non-matmul operation: elementwise negation on encoded tensor
    %negated = arith.negf %encoded : tensor<128xf32, #dynamic_enc_single>
    flow.return %negated : tensor<128xf32, #dynamic_enc_single>
  }
  util.return %0 : tensor<128xf32, #dynamic_enc_single>
}

// CHECK-LABEL: util.func public @non_matmul_dispatch_with_encoding
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<128xf32>, %[[DIM:.+]]: index)
// util.specialize is inserted BEFORE set_encoding for its encoding_dims
// CHECK:         %[[SPEC0:.+]] = util.specialize %[[DIM]]
// CHECK:         iree_encoding.set_encoding %[[ARG0]] encoding_dims{%[[SPEC0]]}
// CHECK:         flow.dispatch.region
// The pass creates iree_encoding.dim ops for the negf operand (not result).
// The dim op is reified through the operand chain back to set_encoding's
// encoding_dims. Inside dispatch, uses are replaced with the specialize result.
// CHECK:           util.specialize %[[SPEC0]]
// CHECK:           arith.negf
// CHECK:           flow.return

// -----

// Test: Multiple tensors with same encoding in dispatch
// Only the first operand with matching encoding triggers specialization.

#dynamic_enc_shared = #iree_encoding.dynamic_layout_test<
  1,
  [[#util<int.assumption.array[<umin = 1, umax = 256>]>, 1],
   [#util<int.assumption.array[<umin = 256>]>, 2]],
  0
>

util.func public @multiple_tensors_same_encoding(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %d0: index, %d1: index) -> tensor<64xf32, #dynamic_enc_shared> {
  // set_encoding outside dispatch for both tensors
  %enc0 = iree_encoding.set_encoding %arg0 encoding_dims{%d0} : tensor<64xf32> -> tensor<64xf32, #dynamic_enc_shared>
  %enc1 = iree_encoding.set_encoding %arg1 encoding_dims{%d1} : tensor<64xf32> -> tensor<64xf32, #dynamic_enc_shared>
  
  // Dispatch returns encoded tensor
  %0 = flow.dispatch.region -> (tensor<64xf32, #dynamic_enc_shared>) {
    %add = arith.addf %enc0, %enc1 : tensor<64xf32, #dynamic_enc_shared>
    flow.return %add : tensor<64xf32, #dynamic_enc_shared>
  }
  util.return %0 : tensor<64xf32, #dynamic_enc_shared>
}

// CHECK-LABEL: util.func public @multiple_tensors_same_encoding
// CHECK-SAME:    (%[[ARG0:.+]]: tensor<64xf32>, %[[ARG1:.+]]: tensor<64xf32>, %[[D0:.+]]: index, %[[D1:.+]]: index)
// util.specialize is inserted BEFORE each set_encoding for its encoding_dims
// CHECK:         %[[SPEC0:.+]] = util.specialize %[[D0]]
// CHECK:         iree_encoding.set_encoding %[[ARG0]] encoding_dims{%[[SPEC0]]}
// CHECK:         %[[SPEC1:.+]] = util.specialize %[[D1]]
// CHECK:         iree_encoding.set_encoding %[[ARG1]] encoding_dims{%[[SPEC1]]}
// CHECK:         flow.dispatch.region
// Inside dispatch, specialize ops reference the outer specialize results
// CHECK:           util.specialize %[[SPEC0]]
// CHECK:           arith.addf
// CHECK:           flow.return

// -----

// Test: Matmul-like dispatch with specialization using EncodingAttr
// For DPS ops like linalg.matmul, getSpecializationOperands returns the
// operand with the encoding. The dim op is reified through to
// set_encoding's encoding_dims, or through tensor.empty's dynamic sizes.

#map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d1)>

// EncodingAttr with dynamic M dimension (iteration_sizes = [?, 64, 64])
#lhs_encoding = #iree_encoding.encoding<
  operand_index = 0 : index,
  op_type = matmul,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_lhs, #map_rhs, #map_out],
  iteration_sizes = [?, 64, 64]>
#rhs_encoding = #iree_encoding.encoding<
  operand_index = 1 : index,
  op_type = matmul,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_lhs, #map_rhs, #map_out],
  iteration_sizes = [?, 64, 64]>
#out_encoding = #iree_encoding.encoding<
  operand_index = 2 : index,
  op_type = matmul,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map_lhs, #map_rhs, #map_out],
  iteration_sizes = [?, 64, 64]>

util.func public @matmul_dispatch_specialization(
    %lhs: tensor<?x64xf32>,
    %rhs: tensor<64x64xf32>,
    %m: index) -> tensor<?x64xf32> {
  
  // Encode tensors outside dispatch with dynamic encoding_dims
  %lhs_enc = iree_encoding.set_encoding %lhs encoding_dims{%m}
    : tensor<?x64xf32> -> tensor<?x64xf32, #lhs_encoding>
  %rhs_enc = iree_encoding.set_encoding %rhs encoding_dims{%m}
    : tensor<64x64xf32> -> tensor<64x64xf32, #rhs_encoding>
  
  // Dispatch with zero initialization inside and unset_encoding at the end
  %result = flow.dispatch.region -> (tensor<?x64xf32>{%m}) {
    // Create zero-initialized output tensor with encoding inside dispatch
    %cst = arith.constant 0.0 : f32
    %empty = tensor.empty(%m) : tensor<?x64xf32, #out_encoding>
    %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x64xf32, #out_encoding>) -> tensor<?x64xf32, #out_encoding>
    
    // Matmul operation - dynamic M dimension
    %matmul = linalg.matmul ins(%lhs_enc, %rhs_enc : tensor<?x64xf32, #lhs_encoding>, tensor<64x64xf32, #rhs_encoding>)
                           outs(%filled : tensor<?x64xf32, #out_encoding>)
                           -> tensor<?x64xf32, #out_encoding>
    
    // Unset encoding before returning
    %decoded = iree_encoding.unset_encoding %matmul encoding_dims{%m}
      : tensor<?x64xf32, #out_encoding> -> tensor<?x64xf32>{%m}
    
    flow.return %decoded : tensor<?x64xf32>
  }
  
  util.return %result : tensor<?x64xf32>
}

// CHECK-LABEL: util.func public @matmul_dispatch_specialization
// CHECK-SAME:    (%[[LHS:.+]]: tensor<?x64xf32>, %[[RHS:.+]]: tensor<64x64xf32>, %[[M:.+]]: index)
// util.specialize is inserted BEFORE the first set_encoding for its encoding_dims
// CHECK:         %[[SPEC0:.+]] = util.specialize %[[M]]
// CHECK:         iree_encoding.set_encoding %[[LHS]] encoding_dims{%[[SPEC0]]}
// CHECK:         iree_encoding.set_encoding %[[RHS]] encoding_dims{%[[SPEC0]]}
// CHECK:         flow.dispatch.region -> (tensor<?x64xf32>{%[[SPEC0]]})
// Inside dispatch: tensor.empty uses the outer specialize result (captured)
// The inner specialize is created for linalg.fill's encoded operand
// CHECK:           tensor.empty(%[[SPEC0]])
// CHECK:           %[[SPEC1:.+]] = util.specialize %[[SPEC0]]
// CHECK:           linalg.fill
// CHECK:           linalg.matmul
// unset_encoding uses the inner specialize result (dominated by it)
// CHECK:           iree_encoding.unset_encoding {{.*}} encoding_dims{%[[SPEC1]]} : {{.*}} -> tensor<?x64xf32>{%[[SPEC1]]}
// CHECK:           flow.return

// -----

// Test: Static encoding dimensions - no specialization needed

#static_enc = #iree_encoding.dynamic_layout_test<
  0,  // no encoding dimensions
  [],
  0
>

util.func public @static_encoding_no_specialization(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %encoded = iree_encoding.set_encoding %arg0 : tensor<128xf32> -> tensor<128xf32, #static_enc>
  %0 = flow.dispatch.region -> (tensor<128xf32>) {
    %unencoded = iree_encoding.unset_encoding %encoded : tensor<128xf32, #static_enc> -> tensor<128xf32>
    flow.return %unencoded : tensor<128xf32>
  }
  util.return %0 : tensor<128xf32>
}

// CHECK-LABEL: util.func public @static_encoding_no_specialization
// CHECK:         iree_encoding.set_encoding
// CHECK:         flow.dispatch.region
// No specialization markers should be inserted for static encodings
// CHECK-NOT:     util.specialize
// CHECK:           iree_encoding.unset_encoding
