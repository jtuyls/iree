// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// Test that encoding_dim folds when encoding_dims are constants.
// The folder traces through the producer chain and folds constant values.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_fold_constant(%arg0: tensor<?x?xf32>) -> (index, index, index) {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c64 = arith.constant 64 : index
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%c128, %c256, %c64} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %dim_m = iree_encoding.encoding_dim %0[0] : tensor<?x?xf32, #encoding>
  %dim_n = iree_encoding.encoding_dim %0[1] : tensor<?x?xf32, #encoding>
  %dim_k = iree_encoding.encoding_dim %0[2] : tensor<?x?xf32, #encoding>
  return %dim_m, %dim_n, %dim_k : index, index, index
}
// CHECK-LABEL: func.func @encoding_dim_fold_constant
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK:         return %[[C128]], %[[C256]], %[[C64]]

// -----

// Test that encoding_dim folds through tensor.cast when encoding_dims are constants.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_fold_through_cast(%arg0: tensor<?x?xf32>) -> (index, index, index) {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c64 = arith.constant 64 : index
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%c128, %c256, %c64} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = tensor.cast %0 : tensor<?x?xf32, #encoding> to tensor<4x8xf32, #encoding>
  %dim_m = iree_encoding.encoding_dim %1[0] : tensor<4x8xf32, #encoding>
  %dim_n = iree_encoding.encoding_dim %1[1] : tensor<4x8xf32, #encoding>
  %dim_k = iree_encoding.encoding_dim %1[2] : tensor<4x8xf32, #encoding>
  return %dim_m, %dim_n, %dim_k : index, index, index
}
// CHECK-LABEL: func.func @encoding_dim_fold_through_cast
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK:         return %[[C128]], %[[C256]], %[[C64]]

// -----

// Test that canonicalization resolves encoding_dim to the dynamic values
// from set_encoding.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_canonicalize_dynamic(%arg0: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> (index, index, index) {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %dim_m = iree_encoding.encoding_dim %0[0] : tensor<?x?xf32, #encoding>
  %dim_n = iree_encoding.encoding_dim %0[1] : tensor<?x?xf32, #encoding>
  %dim_k = iree_encoding.encoding_dim %0[2] : tensor<?x?xf32, #encoding>
  return %dim_m, %dim_n, %dim_k : index, index, index
}
// CHECK-LABEL: func.func @encoding_dim_canonicalize_dynamic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[K:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]], %[[K]]

// -----

// Test that canonicalization traces through tensor.cast for dynamic values.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_canonicalize_through_cast(%arg0: tensor<?x?xf32>, %m: index, %n: index) -> (index, index) {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = tensor.cast %0 : tensor<?x?xf32, #encoding> to tensor<4x8xf32, #encoding>
  %dim_m = iree_encoding.encoding_dim %1[0] : tensor<4x8xf32, #encoding>
  %dim_n = iree_encoding.encoding_dim %1[1] : tensor<4x8xf32, #encoding>
  return %dim_m, %dim_n : index, index
}
// CHECK-LABEL: func.func @encoding_dim_canonicalize_through_cast
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[M:[a-zA-Z0-9]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9]+]]: index
// CHECK:         return %[[M]], %[[N]]

// -----

// Test that encoding_dim is NOT resolved when there's no set_encoding in producer chain.

#encoding = #iree_encoding.testing<>
func.func @encoding_dim_no_set_encoding(%arg0: tensor<?x?xf32, #encoding>) -> index {
  %dim = iree_encoding.encoding_dim %arg0[0] : tensor<?x?xf32, #encoding>
  return %dim : index
}
// CHECK-LABEL: func.func @encoding_dim_no_set_encoding
// CHECK:         %[[DIM:.+]] = iree_encoding.encoding_dim
// CHECK:         return %[[DIM]]
