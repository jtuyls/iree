// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-propagate-encodings))" --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.matmul_k<k_dims = [1]>
util.func public @propagate_encoding_through_collapse_shape(%src: tensor<2x4096x640xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.matmul_k<k_dims = [1]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.matmul_k<k_dims = [2]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]] : tensor<2x4096x640xf16> -> tensor<2x4096x640xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING1]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSED]]

// -----

#encoding = #iree_encoding.matmul_k<k_dims = [1]>
util.func public @propagate_encoding_through_collapse_shape_chain(%src: tensor<2x4096x64x10xf16>) -> tensor<8192x640xf16, #encoding> {
  %collapsed = tensor.collapse_shape %src [[0], [1], [2, 3]] : tensor<2x4096x64x10xf16> into tensor<2x4096x640xf16>
  %collapsed_0 = tensor.collapse_shape %collapsed [[0, 1], [2]] : tensor<2x4096x640xf16> into tensor<8192x640xf16>
  %0 = iree_encoding.set_encoding %collapsed_0 : tensor<8192x640xf16> -> tensor<8192x640xf16, #encoding>
  util.return %0 : tensor<8192x640xf16, #encoding>
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.matmul_k<k_dims = [1]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.matmul_k<k_dims = [2]>
// CHECK-LABEL: @propagate_encoding_through_collapse_shape_chain(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK:         %[[COLLAPSED_0:.+]] = tensor.collapse_shape %[[SRC]] {{\[}}[0], [1], [2, 3]] : tensor<2x4096x64x10xf16> into tensor<2x4096x640xf16>
// CHECK:         %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[COLLAPSED_0]] : tensor<2x4096x640xf16> -> tensor<2x4096x640xf16, #[[$ENCODING1]]>
// CHECK:         %[[COLLAPSED_1:.+]] = tensor.collapse_shape %[[SET_ENCODING]] {{\[}}[0, 1], [2]] : tensor<2x4096x640xf16, #[[$ENCODING1]]> into tensor<8192x640xf16, #[[$ENCODING0]]>
// CHECK:         util.return %[[COLLAPSED_1]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
util.func public @propagate_unset_encoding_through_generic(%arg0 : tensor<?x4096xf32, #encoding>, %arg1 : tensor<f32>, %arg2 : index) -> tensor<?x4096xbf16> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x4096xf32, #encoding> -> tensor<?x4096xf32>{%arg2}
  %1 = tensor.empty(%arg2) : tensor<?x4096xbf16>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %arg1 : tensor<?x4096xf32>, tensor<f32>) outs(%1 : tensor<?x4096xbf16>) {
    ^bb0(%in: f32, %in_1: f32, %out: bf16):
      %1803 = arith.mulf %in, %in_1 : f32
      %1804 = arith.truncf %1803 : f32 to bf16
      linalg.yield %1804 : bf16
    } -> tensor<?x4096xbf16>
  util.return %2 : tensor<?x4096xbf16>
}
// CHECK-LABEL: @propagate_unset_encoding_through_generic2