// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.sign lowers to llvm.call @wrap_sign with signature
//   (state, input_ptr, output_ptr, num_elements, data_type) -> i32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Static 2D i32 input.
  func.func @sign_static_2d_i32(
      %ctx: !hip.context,
      %x: memref<32x16xi32, 1>,
      %y: memref<32x16xi32, 1>) {
    // CHECK-LABEL: llvm.func @sign_static_2d_i32

    hip.sign(%ctx) ins(%x : memref<32x16xi32, 1>)
                   outs(%y : memref<32x16xi32, 1>)

    // CHECK: llvm.call @wrap_sign({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Dynamic 1D f32 input.
  func.func @sign_dynamic_1d_f32(
      %ctx: !hip.context,
      %x: memref<?xf32, 1>,
      %y: memref<?xf32, 1>) {
    // CHECK-LABEL: llvm.func @sign_dynamic_1d_f32

    hip.sign(%ctx) ins(%x : memref<?xf32, 1>)
                   outs(%y : memref<?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_sign({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
