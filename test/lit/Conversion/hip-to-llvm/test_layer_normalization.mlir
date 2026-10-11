// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.layer_norm is correctly lowered to llvm.call @wrap_layer_normalization.
//
// Test cases:
// 1. Static shape: 1x128x4096 f16, with bias, output only
// 2. Dynamic shape: ?x?x512 f16, with bias, output only
// 3. Axis=1 with Mean/InvStdDev output buffers
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // --- Case 1: static shape with bias ---
  func.func @layer_norm_static(
      %ctx: !hip.context,
      %input: memref<1x128x4096xf16, 1>,
      %scale: memref<4096xf16, 1>,
      %bias: memref<4096xf16, 1>,
      %output: memref<1x128x4096xf16, 1>) {
    // CHECK-LABEL: llvm.func @layer_norm_static
    hip.layer_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x128x4096xf16, 1>, memref<4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
    // CHECK: llvm.call @wrap_layer_normalization
    return
  }

  // --- Case 2: dynamic shape with bias ---
  func.func @layer_norm_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf16, 1>,
      %scale: memref<512xf16, 1>,
      %bias: memref<512xf16, 1>,
      %output: memref<?x?x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @layer_norm_dynamic
    hip.layer_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<?x?x512xf16, 1>, memref<512xf16, 1>, memref<512xf16, 1>)
        outs(%output : memref<?x?x512xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_layer_normalization
    return
  }

  // --- Case 3: optional stats outputs and non-default axis ---
  func.func @layer_norm_stats(
      %ctx: !hip.context,
      %input: memref<2x3x4xf16, 1>,
      %scale: memref<3x4xf16, 1>,
      %output: memref<2x3x4xf16, 1>,
      %mean: memref<2x1x1xf32, 1>,
      %inv_std: memref<2x1x1xf32, 1>) {
    // CHECK-LABEL: llvm.func @layer_norm_stats
    hip.layer_norm(%ctx)
        ins(%input, %scale : memref<2x3x4xf16, 1>, memref<3x4xf16, 1>)
        outs(%output, %mean, %inv_std :
             memref<2x3x4xf16, 1>, memref<2x1x1xf32, 1>,
             memref<2x1x1xf32, 1>)
        {axis = 1 : i64, epsilon = 9.99999974E-6 : f32,
         stash_type = 1 : i64}
    // CHECK: llvm.call @wrap_layer_normalization
    return
  }
}
