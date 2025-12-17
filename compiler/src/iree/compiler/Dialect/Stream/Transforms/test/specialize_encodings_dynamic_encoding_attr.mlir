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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>

// RHS encoding
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>

// Result encoding
#encoding_result = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>

util.global private @device = #device_target
util.func public @matmul_encoding_dynamic_specialization(%d0: index, %d1: index, %d2: index) -> (index, index, index) {
  %size_lhs = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x?xf32, #encoding_lhs>{%d0, %d2} : index
  %size_rhs = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x?xf32, #encoding_rhs>{%d2, %d1} : index
  %size_result = stream.tensor.sizeof on(#hal.device.affinity<@device>) tensor<?x?xf32, #encoding_result>{%d0, %d1} : index
  util.return %size_lhs, %size_rhs, %size_result : index, index, index
}

// When dynamic specialization is enabled for EncodingAttr:
// - supportsSpecialization() returns true (because iteration_sizes is present)
// - getSpecializationInfo() returns just a fallback encoding (no variants yet)
// - The resolver resolves the fallback to a layout
// - Result is SpecializableLayoutAttr with empty variants and fallback layout

// CHECK-LABEL: util.func public @matmul_encoding_dynamic_specialization
// CHECK-SAME:    (%[[D0:.+]]: index, %[[D1:.+]]: index, %[[D2:.+]]: index)
//
// LHS tensor: ?x? with encoding -> specializable_layout with 3 encoding dims
// CHECK:         %[[SIZE_LHS:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x?xf32, #iree_encoding.specializable_layout<3, {{\[\[}}{{\]\]}}, {{\[\[}}{{\]\]}}, #iree_encoding.specialized<42, tensor<?x?xf32>>>>
// CHECK-SAME:      {%[[D0]], %[[D2]]} : index
//
// RHS tensor: ?x? with encoding -> specializable_layout with 3 encoding dims
// CHECK:         %[[SIZE_RHS:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x?xf32, #iree_encoding.specializable_layout<3, {{\[\[}}{{\]\]}}, {{\[\[}}{{\]\]}}, #iree_encoding.specialized<42, tensor<?x?xf32>>>>
// CHECK-SAME:      {%[[D2]], %[[D1]]} : index
//
// Result tensor: ?x? with encoding -> specializable_layout with 3 encoding dims
// CHECK:         %[[SIZE_RES:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device>)
// CHECK-SAME:      tensor<?x?xf32, #iree_encoding.specializable_layout<3, {{\[\[}}{{\]\]}}, {{\[\[}}{{\]\]}}, #iree_encoding.specialized<42, tensor<?x?xf32>>>>
// CHECK-SAME:      {%[[D0]], %[[D1]]} : index
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
// This tests that EncodingAttr works with the real GPU encoding resolver.
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

// Matmul LHS encoding with iteration_sizes for dynamic specialization
#encoding_gpu_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, ?, ?]>

util.global private @device_gpu = #device_target_gpu
util.func public @gpu_encoding_dynamic_specialization(%d0: index, %d1: index) -> index {
  %size = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>) tensor<?x?xf32, #encoding_gpu_lhs>{%d0, %d1} : index
  util.return %size : index
}

// With GPU resolver and dynamic specialization enabled, the EncodingAttr gets:
// - Converted to SpecializableLayoutAttr (because iteration_sizes is present)
// - The fallback layout is resolved by the GPU resolver to a layout with encoding_info
// CHECK-LABEL: util.func public @gpu_encoding_dynamic_specialization
// CHECK-SAME:    (%[[D0:.+]]: index, %[[D1:.+]]: index)
// CHECK:         %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>)
// The encoding should be specializable_layout with 3 dims, empty variants, and GPU-resolved fallback
// CHECK-SAME:      tensor<?x?xf32, #iree_encoding.specializable_layout<3, {{\[\[}}{{\]\]}}, {{\[\[}}{{\]\]}},
// The fallback layout contains GPU resolver info with encoding_info
// CHECK-SAME:        #iree_gpu.gpu_encoding_resolver
// CHECK-SAME:        encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-SAME:      >{%[[D0]], %[[D1]]} : index
// CHECK:         util.return %[[SIZE]]

// -----

// Test GPU resolver WITHOUT iteration_sizes (no dynamic specialization).

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

// Encoding WITHOUT iteration_sizes - should get regular layout, not specializable_layout
#encoding_gpu_no_iter = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>

util.global private @device_gpu = #device_target_gpu
util.func public @gpu_encoding_no_specialization(%d0: index, %d1: index) -> index {
  %size = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>) tensor<?x?xf32, #encoding_gpu_no_iter>{%d0, %d1} : index
  util.return %size : index
}

// Without iteration_sizes, encoding gets a regular layout (not specializable_layout)
// CHECK:       #[[$GPU_LAYOUT:.+]] = #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<{{.+}}encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]{{.+}}}}>]>
// CHECK-LABEL: util.func public @gpu_encoding_no_specialization
// CHECK-SAME:    (%[[D0:.+]]: index, %[[D1:.+]]: index)
// CHECK:         %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.affinity<@device_gpu>)
// CHECK-SAME:      tensor<?x?xf32, #[[$GPU_LAYOUT]]>{%[[D0]], %[[D1]]} : index
// CHECK:         util.return %[[SIZE]]
