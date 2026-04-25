// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.resize -> llvm.call @wrap_resize lowering.  Covers the Kokoro
// vocoder pattern: NCHW with H=1, large W upsample, mode=nearest,
// coord_transform=half_pixel, fp16 element type.
//
// Expected wrap_resize signature:
//   wrap_resize(state, input, output,
//               in_shape*, in_strides*, out_shape*, out_strides*,
//               rank, data_type, mode, coord_xform, cubic_coeff_a)
//   -> i32
// 12 args total: 7 ptrs + 4 i64 + 1 f32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Kokoro-style: NCHW, H=1, W upsample, fp16, nearest + half_pixel.
  func.func @resize_nearest_half_pixel_nchw_fp16(
      %ctx: !hip.context,
      %input: memref<1x32x1x100xf16, 1>,
      %output: memref<1x32x1x200xf16, 1>) {
    // CHECK-LABEL: llvm.func @resize_nearest_half_pixel_nchw_fp16
    hip.resize(%ctx) ins(%input : memref<1x32x1x100xf16, 1>)
                     outs(%output : memref<1x32x1x200xf16, 1>)
                     {mode = 0 : i64, coord_transform = 0 : i64,
                      cubic_coeff_a = -7.500000e-01 : f32}
    // CHECK: llvm.call @wrap_resize({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
    return
  }

  // Linear (bilinear) resize over a 2-D NHWC image, fp32.
  func.func @resize_linear_align_corners_fp32(
      %ctx: !hip.context,
      %input: memref<1x4x16x16xf32, 1>,
      %output: memref<1x4x32x32xf32, 1>) {
    // CHECK-LABEL: llvm.func @resize_linear_align_corners_fp32
    hip.resize(%ctx) ins(%input : memref<1x4x16x16xf32, 1>)
                     outs(%output : memref<1x4x32x32xf32, 1>)
                     {mode = 1 : i64, coord_transform = 2 : i64,
                      cubic_coeff_a = -7.500000e-01 : f32}
    // CHECK: llvm.call @wrap_resize({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
    return
  }

  // Cubic resize, bf16, asymmetric coord transform.
  func.func @resize_cubic_asymmetric_bf16(
      %ctx: !hip.context,
      %input: memref<1x8x12x12xbf16, 1>,
      %output: memref<1x8x24x24xbf16, 1>) {
    // CHECK-LABEL: llvm.func @resize_cubic_asymmetric_bf16
    hip.resize(%ctx) ins(%input : memref<1x8x12x12xbf16, 1>)
                     outs(%output : memref<1x8x24x24xbf16, 1>)
                     {mode = 2 : i64, coord_transform = 3 : i64,
                      cubic_coeff_a = -5.000000e-01 : f32}
    // CHECK: llvm.call @wrap_resize({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
    return
  }
}
