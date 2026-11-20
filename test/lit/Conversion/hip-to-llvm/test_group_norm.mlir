// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify hip.group_norm lowers to llvm.call @wrap_group_norm.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @group_norm_nchw(
      %ctx: !hip.context,
      %input: memref<1x8x4x4xf32, 1>,
      %scale: memref<8xf32, 1>,
      %bias: memref<8xf32, 1>,
      %output: memref<1x8x4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @group_norm_nchw
    hip.group_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x8x4x4xf32, 1>, memref<8xf32, 1>, memref<8xf32, 1>)
        outs(%output : memref<1x8x4x4xf32, 1>)
        {activation = 1 : i64, channels_last = 0 : i64, groups = 4 : i64}
    // CHECK: llvm.mlir.constant(4 : i64)
    // CHECK: llvm.call @wrap_group_norm
    return
  }

  func.func @group_norm_nhwc_dynamic(
      %ctx: !hip.context,
      %input: memref<1x?x?x8xf16, 1>,
      %scale: memref<8xf16, 1>,
      %bias: memref<8xf16, 1>,
      %output: memref<1x?x?x8xf16, 1>) {
    // CHECK-LABEL: llvm.func @group_norm_nhwc_dynamic
    hip.group_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x?x?x8xf16, 1>, memref<8xf16, 1>, memref<8xf16, 1>)
        outs(%output : memref<1x?x?x8xf16, 1>)
        {activation = 0 : i64, channels_last = 1 : i64, groups = 2 : i64}
    // CHECK: llvm.extractvalue
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_group_norm
    return
  }

  // A residual is passed as the fifth argument; alpha precedes epsilon.
  func.func @group_norm_residual_leaky(
      %ctx: !hip.context,
      %input: memref<1x8x4x4xf32, 1>,
      %scale: memref<8xf32, 1>,
      %bias: memref<8xf32, 1>,
      %residual: memref<1x8x4x4xf32, 1>,
      %output: memref<1x8x4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @group_norm_residual_leaky
    hip.group_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x8x4x4xf32, 1>, memref<8xf32, 1>, memref<8xf32, 1>)
        residual(%residual : memref<1x8x4x4xf32, 1>)
        outs(%output : memref<1x8x4x4xf32, 1>)
        {activation = 3 : i64, alpha = 2.000000e-01 : f32,
         channels_last = 0 : i64, groups = 8 : i64}
    // CHECK-DAG: %[[ALPHA:.*]] = llvm.mlir.constant(2.000000e-01 : f32) : f32
    // CHECK-DAG: %[[ACT:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: llvm.call @wrap_group_norm(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[RES:[^,]+]], %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[ACT]], %{{[^,]+}}, %[[ALPHA]], %{{[^,]+}})
    // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, f32, f32) -> i32
    return
  }

  // Without a residual the fifth argument is a null pointer.
  func.func @group_norm_no_residual(
      %ctx: !hip.context,
      %input: memref<1x8x4x4xf32, 1>,
      %scale: memref<8xf32, 1>,
      %bias: memref<8xf32, 1>,
      %output: memref<1x8x4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @group_norm_no_residual
    hip.group_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x8x4x4xf32, 1>, memref<8xf32, 1>, memref<8xf32, 1>)
        outs(%output : memref<1x8x4x4xf32, 1>)
        {activation = 2 : i64, channels_last = 0 : i64, groups = 8 : i64}
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @wrap_group_norm(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[NULL]],
    return
  }
}
