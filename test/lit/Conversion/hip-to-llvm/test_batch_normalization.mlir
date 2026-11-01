// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.batch_norm lowers to llvm.call @wrap_batch_normalization.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @batch_norm_static(
      %ctx: !hip.context,
      %input: memref<1x4x1x1xf16, 1>,
      %scale: memref<4xf16, 1>,
      %bias: memref<4xf16, 1>,
      %mean: memref<4xf16, 1>,
      %variance: memref<4xf16, 1>,
      %output: memref<1x4x1x1xf16, 1>) {
    // CHECK-LABEL: llvm.func @batch_norm_static
    hip.batch_norm(%ctx)
        ins(%input, %scale, %bias, %mean, %variance :
            memref<1x4x1x1xf16, 1>, memref<4xf16, 1>, memref<4xf16, 1>,
            memref<4xf16, 1>, memref<4xf16, 1>)
        outs(%output : memref<1x4x1x1xf16, 1>)
        {epsilon = 1.001000e-05 : f32}
    // CHECK: llvm.call @wrap_batch_normalization
    return
  }

  func.func @batch_norm_dynamic(
      %ctx: !hip.context,
      %input: memref<1x3x?x?xf32, 1>,
      %scale: memref<3xf32, 1>,
      %bias: memref<3xf32, 1>,
      %mean: memref<3xf32, 1>,
      %variance: memref<3xf32, 1>,
      %output: memref<1x3x?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @batch_norm_dynamic
    hip.batch_norm(%ctx)
        ins(%input, %scale, %bias, %mean, %variance :
            memref<1x3x?x?xf32, 1>, memref<3xf32, 1>, memref<3xf32, 1>,
            memref<3xf32, 1>, memref<3xf32, 1>)
        outs(%output : memref<1x3x?x?xf32, 1>)
        {epsilon = 1.000000e-05 : f32}
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_batch_normalization
    return
  }
}
