// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A float matmul with a bias has no fused kernel, and must not silently lower
// to the bias-free float matmul (which would drop the bias).

// RUN: not hip-mlir-opt %s --convert-hip-to-llvm 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'hip.matmul'
func.func @float_matmul_bias(%ctx: !hip.context, %A: memref<4x8xf32, 1>,
                             %B: memref<8x6xf32, 1>, %C: memref<6xf32, 1>,
                             %Y: memref<4x6xf32, 1>) {
  hip.matmul(%ctx) ins(%A, %B, %C : memref<4x8xf32, 1>, memref<8x6xf32, 1>, memref<6xf32, 1>)
      outs(%Y : memref<4x6xf32, 1>) {quant_types = [f32, f32, f32, f32]}
  return
}
