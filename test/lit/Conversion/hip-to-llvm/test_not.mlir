// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.not lowers to llvm.call @wrap_not with signature
//   (state, input_ptr, output_ptr, num_elements, data_type) -> i32.
// Inputs are i1 (boolean). Since i1 is not in the HIPDNN dtype enum, the
// lowering passes 0 as a sentinel (see UnaryElementwiseLowering.cpp).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Static 1D boolean input.
  func.func @not_static_1d(
      %ctx: !hip.context,
      %x: memref<64xi1, 1>,
      %y: memref<64xi1, 1>) {
    // CHECK-LABEL: llvm.func @not_static_1d

    hip.not(%ctx) ins(%x : memref<64xi1, 1>)
                  outs(%y : memref<64xi1, 1>)

    // CHECK: llvm.call @wrap_not({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Dynamic 2D boolean input.
  func.func @not_dynamic_2d(
      %ctx: !hip.context,
      %x: memref<?x?xi1, 1>,
      %y: memref<?x?xi1, 1>) {
    // CHECK-LABEL: llvm.func @not_dynamic_2d

    hip.not(%ctx) ins(%x : memref<?x?xi1, 1>)
                  outs(%y : memref<?x?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_not({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
