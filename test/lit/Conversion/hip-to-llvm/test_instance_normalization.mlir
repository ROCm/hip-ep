// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.instance_norm lowers to llvm.call @wrap_instance_normalization.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @instance_norm_static(
      %ctx: !hip.context,
      %input: memref<2x3x8x8xf32, 1>,
      %scale: memref<3xf32, 1>,
      %bias: memref<3xf32, 1>,
      %output: memref<2x3x8x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @instance_norm_static
    hip.instance_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<2x3x8x8xf32, 1>, memref<3xf32, 1>, memref<3xf32, 1>)
        outs(%output : memref<2x3x8x8xf32, 1>)
        {epsilon = 1.000000e-05 : f32}
    // CHECK: llvm.call @wrap_instance_normalization
    return
  }

  func.func @instance_norm_dynamic(
      %ctx: !hip.context,
      %input: memref<1x3x?x?xf16, 1>,
      %scale: memref<3xf16, 1>,
      %bias: memref<3xf16, 1>,
      %output: memref<1x3x?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @instance_norm_dynamic
    hip.instance_norm(%ctx)
        ins(%input, %scale, %bias :
            memref<1x3x?x?xf16, 1>, memref<3xf16, 1>, memref<3xf16, 1>)
        outs(%output : memref<1x3x?x?xf16, 1>)
        {epsilon = 1.000000e-05 : f32}
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 2]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_instance_normalization
    return
  }
}
