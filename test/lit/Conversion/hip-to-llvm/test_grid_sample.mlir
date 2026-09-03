// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.grid_sample lowers to llvm.call @wrap_grid_sample.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @grid_sample_static(
      %ctx: !hip.context,
      %input: memref<1x3x8x8xf32, 1>,
      %grid: memref<1x4x4x2xf32, 1>,
      %output: memref<1x3x4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @grid_sample_static
    hip.grid_sample(%ctx)
        ins(%input, %grid : memref<1x3x8x8xf32, 1>, memref<1x4x4x2xf32, 1>)
        outs(%output : memref<1x3x4x4xf32, 1>)
        {mode = 1, padding_mode = 0, align_corners = 0}
    // CHECK: llvm.call @wrap_grid_sample
    return
  }

  func.func @grid_sample_dynamic(
      %ctx: !hip.context,
      %input: memref<1x3x?x?xf32, 1>,
      %grid: memref<1x?x?x2xf32, 1>,
      %output: memref<1x3x?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @grid_sample_dynamic
    hip.grid_sample(%ctx)
        ins(%input, %grid : memref<1x3x?x?xf32, 1>, memref<1x?x?x2xf32, 1>)
        outs(%output : memref<1x3x?x?xf32, 1>)
        {mode = 1, padding_mode = 0, align_corners = 0}
    // CHECK: llvm.call @wrap_grid_sample
    return
  }
}
