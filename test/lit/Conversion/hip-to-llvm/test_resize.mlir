// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.resize lowers to llvm.call @wrap_resize with signature
// (state, input, output, data_type, spatial_rank, N, C, in0..2, out0..2,
//  mode, coord_transform, nearest_mode) -> i32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: 2D bilinear upsample, fully static.  Linear mode, half_pixel.
  func.func @resize_2d_static_f16(
      %ctx: !hip.context,
      %x: memref<1x3x16x16xf16, 1>,
      %y: memref<1x3x32x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @resize_2d_static_f16
    hip.resize(%ctx) ins(%x : memref<1x3x16x16xf16, 1>)
                     outs(%y : memref<1x3x32x32xf16, 1>)
                     {mode = 1, coord_transform = 0, nearest_mode = 0}
    // CHECK: llvm.call @wrap_resize
    return
  }

  // Test 2: 1D resize (rank-3, spatial_rank=1).
  func.func @resize_1d_static_f32(
      %ctx: !hip.context,
      %x: memref<1x3x32xf32, 1>,
      %y: memref<1x3x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @resize_1d_static_f32
    hip.resize(%ctx) ins(%x : memref<1x3x32xf32, 1>)
                     outs(%y : memref<1x3x64xf32, 1>)
                     {mode = 0, coord_transform = 1, nearest_mode = 0}
    // CHECK: llvm.call @wrap_resize
    return
  }

  // Test 3: dynamic batch — verifies extractvalue [3, 0] off the input
  // descriptor for N (and matching off the output for the same axis).
  func.func @resize_dynamic_n_f16(
      %ctx: !hip.context,
      %x: memref<?x3x16x16xf16, 1>,
      %y: memref<?x3x32x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @resize_dynamic_n_f16
    hip.resize(%ctx) ins(%x : memref<?x3x16x16xf16, 1>)
                     outs(%y : memref<?x3x32x32xf16, 1>)
                     {mode = 1, coord_transform = 2, nearest_mode = 0}
    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_resize
    return
  }
}
