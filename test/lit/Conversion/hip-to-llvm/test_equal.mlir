// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.equal lowers to llvm.call @wrap_equal with signature
//   (state, lhs, rhs, output, num_elements, data_type) -> i32.
// Output is i1 (bool, 1 byte); data_type identifies the input element type.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Static i64 inputs.
  func.func @equal_static_i64(
      %ctx: !hip.context,
      %a: memref<8xi64, 1>,
      %b: memref<8xi64, 1>,
      %c: memref<8xi1, 1>) {
    // CHECK-LABEL: llvm.func @equal_static_i64

    hip.equal(%ctx) ins(%a, %b : memref<8xi64, 1>, memref<8xi64, 1>)
                    outs(%c : memref<8xi1, 1>)

    // CHECK: llvm.call @wrap_equal({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    return
  }

  // Dynamic f32 inputs.
  func.func @equal_dynamic_f32(
      %ctx: !hip.context,
      %a: memref<?xf32, 1>,
      %b: memref<?xf32, 1>,
      %c: memref<?xi1, 1>) {
    // CHECK-LABEL: llvm.func @equal_dynamic_f32

    hip.equal(%ctx) ins(%a, %b : memref<?xf32, 1>, memref<?xf32, 1>)
                    outs(%c : memref<?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_equal({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    return
  }
}
