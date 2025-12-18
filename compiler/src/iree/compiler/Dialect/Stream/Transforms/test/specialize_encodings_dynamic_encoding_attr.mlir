// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-stream-specialize-encodings)' \
// RUN:   --iree-encoding-enable-dynamic-specialization=true \
// RUN:   --verify-diagnostics %s | FileCheck %s

//------------------------------------------------------------------------------
// EncodingAttr dynamic layout specialization tests.
// When the dynamic specialization flag is enabled, EncodingAttr implements
// DynamicLayoutSpecializerAttr and gets converted to SpecializableLayoutAttr.
//------------------------------------------------------------------------------

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_encoding.specialization_resolver<42>}>
#device_target = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target]> : !hal.device

// LHS encoding with matmul user indexing maps and iteration_sizes
// Only M dimension is dynamic, N=1024 and K=512 are static.
// This triggers specialization (exactly 1 dynamic dimension).
#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 1024, 512]>

// RHS encoding
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 1024, 512]>

// Result encoding
#encoding_result = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 1024, 512]>

util.global private @device = #device_target
util.func public @matmul_encoding_dynamic_specialization(%d0: index) -> (index, index, index) {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %size_lhs = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x512xf32, #encoding_lhs>{%d0} : index
  %size_rhs = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<512x1024xf32, #encoding_rhs> : index
  %size_result = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x1024xf32, #encoding_result>{%d0} : index
  util.return %size_lhs, %size_rhs, %size_result : index, index, index
}

// When dynamic specialization is enabled for EncodingAttr with exactly 1 dynamic dimension:
// - supportsSpecialization() returns true
// - getSpecializationInfo() returns 1 variant (for M>=512) + fallback
// - Each gets resolved to a layout
// - Result is SpecializableLayoutAttr with 1 variant and fallback layout

// CHECK-LABEL: util.func public @matmul_encoding_dynamic_specialization
// CHECK-SAME:    (%[[D0:.+]]: index)
//
// LHS tensor: ?x512 with encoding -> specializable_layout with 1 encoding dim
// The variant layout has ranges <umin = 512, udiv = 256>
// CHECK:         %[[SIZE_LHS:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x512xf32, #iree_encoding.specializable_layout<1,
// CHECK-SAME:        {{\[\[}}#util<int.assumption.array[<umin = 512, udiv = 256>]>{{\]\]}}
// CHECK-SAME:        {%[[D0]]} : index
//
// RHS tensor: 512x1024 (fully static, no dynamic dims to specialize on)
// CHECK:         %[[SIZE_RHS:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<512x1024xf32, #iree_encoding.specializable_layout<1,
//
// Result tensor: ?x1024 with encoding -> specializable_layout with 1 encoding dim
// CHECK:         %[[SIZE_RES:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x1024xf32, #iree_encoding.specializable_layout<1,
// CHECK-SAME:        {{\[\[}}#util<int.assumption.array[<umin = 512, udiv = 256>]>{{\]\]}}
// CHECK-SAME:        {%[[D0]]} : index
//
// CHECK:         util.return %[[SIZE_LHS]], %[[SIZE_RHS]], %[[SIZE_RES]]

// -----

// Test that EncodingAttr WITHOUT iteration_sizes does not get specialized.

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_encoding.specialization_resolver<42>}>
#device_target = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target]> : !hal.device

// Encoding WITHOUT iteration_sizes should NOT be specialized
#encoding_no_iter_sizes = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>

util.global private @device = #device_target
util.func public @encoding_without_iteration_sizes(%d0: index, %d1: index) -> index {
  %size = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x?xf32, #encoding_no_iter_sizes>{%d0, %d1} : index
  util.return %size : index
}

// Without iteration_sizes, EncodingAttr::supportsSpecialization() returns false,
// so the encoding follows the standard path and gets a regular layout (not specializable_layout).
// CHECK-DAG:   #[[$LAYOUT:.+]] = #iree_encoding.layout<[#iree_encoding.specialized<42, tensor<?x?xf32>>]>
// CHECK-LABEL: util.func public @encoding_without_iteration_sizes
// CHECK-SAME:    (%[[D0:.+]]: index, %[[D1:.+]]: index)
// CHECK:         %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x?xf32, #[[$LAYOUT]]>{%[[D0]], %[[D1]]} : index
// CHECK:         util.return %[[SIZE]]

// -----

//------------------------------------------------------------------------------
// GPU resolver test with EncodingAttr dynamic specialization.
// This tests that EncodingAttr with exactly 1 dynamic dimension gets specialized.
//------------------------------------------------------------------------------

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>,
    iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
                                       features = "",
                                       wgp = <compute = fp32,
                                              storage =  b32,
                                              subgroup =  none,
                                              mma = [<MFMA_F32_16x16x4_F32>],
                                              subgroup_size_choices = [64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647],
                                              max_load_instruction_bits = 128,
                                              simds_per_wgp = 4,
                                              vgpr_space_bits = 16384>>
  }>
#device_target_gpu = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm]> : !hal.device

// Matmul LHS encoding with exactly 1 dynamic dimension (M) for specialization
#encoding_gpu_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 1024, 512]>

util.global private @device_gpu = #device_target_gpu
util.func public @gpu_encoding_dynamic_specialization(%d0: index) -> index {
  %size = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>) tensor<?x512xf32, #encoding_gpu_lhs>{%d0} : index
  util.return %size : index
}

// With GPU resolver and exactly 1 dynamic dimension:
// - Gets converted to SpecializableLayoutAttr with 1 variant and fallback
// - The variant has range <umin = 512, udiv = 256>
// - Both variant and fallback are resolved by the GPU resolver
// - IMPORTANTLY: The variant and fallback have DIFFERENT tile sizes!
//   Variant (M=512): innerTileSizes = [32, 16]
//   Fallback (M=?):  innerTileSizes = [128, 16]
// CHECK-LABEL: util.func public @gpu_encoding_dynamic_specialization
// CHECK-SAME:    (%[[D0:.+]]: index)
// CHECK:         %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>)
// The encoding should be specializable_layout with 1 encoding dim
// CHECK-SAME:      tensor<?x512xf32, #iree_encoding.specializable_layout<1,
// The variant ranges: M >= 512, divisible by 256
// CHECK-SAME:        {{\[\[}}#util<int.assumption.array[<umin = 512, udiv = 256>]>{{\]\]}}
// The variant layout (GPU resolver) - smaller tiles for M=512
// CHECK-SAME:        {{\[\[}}#iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [32, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = {{\[\[}}{{\[}}"CrossIntrinsic", 2 : i16{{\]}}, {{\[}}"CrossThread", 16 : i16{{\]\]}}, {{\[\[}}"CrossIntrinsic", 4 : i16{{\]}}, {{\[}}"CrossThread", 4 : i16{{\]\]\]}}, permutation = [0, 3, 1, 2]}}}>
// CHECK-SAME:        {{\]\]}}
// The fallback layout (GPU resolver) - larger tiles for dynamic M
// CHECK-SAME:        #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [128, 16], outerDimsPerm = [0, 1], swizzle = {expandShape = {{\[\[}}{{\[}}"CrossThread", 2 : i16{{\]}}, {{\[}}"CrossIntrinsic", 4 : i16{{\]}}, {{\[}}"CrossThread", 16 : i16{{\]\]}}, {{\[\[}}"CrossIntrinsic", 4 : i16{{\]}}, {{\[}}"CrossThread", 4 : i16{{\]\]\]}}, permutation = [0, 1, 4, 2, 3]}}}>
// CHECK-SAME:        >>
// CHECK-SAME:      {%[[D0]]} : index
// CHECK:         util.return %[[SIZE]]

// -----

// Test GPU resolver with 3 dynamic dimensions (not supported, falls back to regular layout).
// Currently only exactly 1 dynamic dimension is supported for specialization.

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>,
    iree_codegen.target_info = #iree_gpu.target<arch = "gfx942",
                                       features = "",
                                       wgp = <compute = fp32,
                                              storage =  b32,
                                              subgroup =  none,
                                              mma = [<MFMA_F32_16x16x4_F32>],
                                              subgroup_size_choices = [64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647],
                                              max_load_instruction_bits = 128,
                                              simds_per_wgp = 4,
                                              vgpr_space_bits = 16384>>
  }>
#device_target_gpu = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm]> : !hal.device

// Encoding with 3 dynamic dimensions - doesn't get specialized (only 1 dynamic dim supported)
#encoding_gpu_3_dyn = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>

util.global private @device_gpu = #device_target_gpu
util.func public @gpu_encoding_no_specialization(%d0: index, %d1: index) -> index {
  %size = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>) tensor<?x?xf32, #encoding_gpu_3_dyn>{%d0, %d1} : index
  util.return %size : index
}

// With 3 dynamic dimensions, supportsSpecialization() returns false, so it gets a regular layout
// CHECK-DAG:   #[[$GPU_LAYOUT:.+]] = #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<{{.+}}encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]{{.+}}}}>]>
// CHECK-LABEL: util.func public @gpu_encoding_no_specialization
// CHECK-SAME:    (%[[D0:.+]]: index, %[[D1:.+]]: index)
// CHECK:         %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>)
// CHECK-SAME:      tensor<?x?xf32, #[[$GPU_LAYOUT]]>{%[[D0]], %[[D1]]} : index
// CHECK:         util.return %[[SIZE]]
