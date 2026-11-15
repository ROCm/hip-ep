// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.roi_align lowers to llvm.call @wrap_roi_align, including the
// f32 spatial_scale argument and dynamic num_rois.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @roi_align_static(
      %ctx: !hip.context,
      %input: memref<1x3x8x8xf32, 1>,
      %rois: memref<4x4xf32, 1>,
      %idx: memref<4xi64, 1>,
      %output: memref<4x3x2x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @roi_align_static
    // CHECK: llvm.mlir.constant(5.000000e-01 : f32) : f32
    hip.roi_align(%ctx)
        ins(%input, %rois, %idx : memref<1x3x8x8xf32, 1>, memref<4x4xf32, 1>, memref<4xi64, 1>)
        outs(%output : memref<4x3x2x2xf32, 1>)
        {mode = 0, output_height = 2, output_width = 2, sampling_ratio = 2,
         spatial_scale = 5.000000e-01 : f32, half_pixel = 1}
    // CHECK: llvm.call @wrap_roi_align
    return
  }

  func.func @roi_align_dynamic(
      %ctx: !hip.context,
      %input: memref<1x256x?x?xf16, 1>,
      %rois: memref<?x4xf16, 1>,
      %idx: memref<?xi64, 1>,
      %output: memref<?x256x7x7xf16, 1>) {
    // CHECK-LABEL: llvm.func @roi_align_dynamic
    hip.roi_align(%ctx)
        ins(%input, %rois, %idx : memref<1x256x?x?xf16, 1>, memref<?x4xf16, 1>, memref<?xi64, 1>)
        outs(%output : memref<?x256x7x7xf16, 1>)
        {mode = 1, output_height = 7, output_width = 7, sampling_ratio = 0,
         spatial_scale = 2.500000e-01 : f32, half_pixel = 0}
    // CHECK: llvm.call @wrap_roi_align
    return
  }
}
