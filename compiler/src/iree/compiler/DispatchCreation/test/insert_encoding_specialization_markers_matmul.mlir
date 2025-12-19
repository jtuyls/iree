// RUN: iree-opt --split-input-file \
// RUN:   --iree-encoding-enable-dynamic-specialization=true \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-insert-encoding-specialization-markers))' %s \
// RUN:   | FileCheck %s

// Test encoding_dim reification through tensor.empty and linalg ops in a matmul dispatch region

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [?, 128256, 4096]>

#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [?, 128256, 4096]>

#encoding_result = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f16, f16, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iteration_sizes = [?, 128256, 4096]>

util.func public @matmul_dispatch_region(
    %lhs: tensor<?x4096xf16, #encoding_lhs>,
    %rhs: tensor<128256x4096xf16, #encoding_rhs>) -> tensor<?x128256xf32> {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128256 = arith.constant 128256 : index
  %dim0 = tensor.dim %lhs, %c0 : tensor<?x4096xf16, #encoding_lhs>
  
  %result = flow.dispatch.region -> (tensor<?x128256xf32>{%dim0}) {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty(%dim0) : tensor<?x128256xf32, #encoding_result>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x128256xf32, #encoding_result>) 
        -> tensor<?x128256xf32, #encoding_result>
    %matmul = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%lhs, %rhs : tensor<?x4096xf16, #encoding_lhs>, tensor<128256x4096xf16, #encoding_rhs>)
        outs(%fill : tensor<?x128256xf32, #encoding_result>) {
      ^bb0(%in: f16, %in_0: f16, %out: f32):
        %0 = arith.extf %in : f16 to f32
        %1 = arith.extf %in_0 : f16 to f32
        %2 = arith.mulf %0, %1 : f32
        %3 = arith.addf %out, %2 : f32
        linalg.yield %3 : f32
      } -> tensor<?x128256xf32, #encoding_result>
    %unencoded = iree_encoding.unset_encoding %matmul encoding_dims{%dim0, %c128256, %c4096}
        : tensor<?x128256xf32, #encoding_result> -> tensor<?x128256xf32>{%dim0}
    flow.return %unencoded : tensor<?x128256xf32>
  }
  util.return %result : tensor<?x128256xf32>
}

// CHECK-LABEL: util.func public @matmul_dispatch_region
// CHECK:         %[[DIM0:.+]] = tensor.dim
// CHECK:         flow.dispatch.region
// CHECK:           %[[EMPTY:.+]] = tensor.empty(%[[DIM0]])
// The marker insertion pass uses getSpecializationOperands to find operands to
// specialize, creates encoding_dim ops, runs reification patterns to fold them
// to %dim0, then deduplicates util.specialize ops. The result is one marker
// per unique reified value.
// CHECK:           util.specialize %[[DIM0]] : index
// CHECK:           %[[FILL:.+]] = linalg.fill {{.+}} outs(%[[EMPTY]]
// CHECK:           %[[MATMUL:.+]] = linalg.generic {{.+}} outs(%[[FILL]]
// CHECK:           iree_encoding.unset_encoding %[[MATMUL]]

