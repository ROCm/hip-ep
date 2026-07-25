// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.less lowers to llvm.call @wrap_less with signature
//   (state, lhs, rhs, output,
//    lhs_n..lhs_w, rhs_n..rhs_w, out_n..out_w, data_type) -> i32.
// Full 4D operand shapes are passed so the runtime can materialise ONNX
// broadcast via hip_expand before the flat comparison kernel.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @less_static_f32(
      %ctx: !hip.context,
      %a: memref<4x8xf32, 1>,
      %b: memref<4x8xf32, 1>,
      %c: memref<4x8xi1, 1>) {
    // CHECK-LABEL: llvm.func @less_static_f32

    hip.less(%ctx) ins(%a, %b : memref<4x8xf32, 1>, memref<4x8xf32, 1>)
                   outs(%c : memref<4x8xi1, 1>)

    // CHECK: llvm.call @wrap_less({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  func.func @less_dynamic_i32(
      %ctx: !hip.context,
      %a: memref<?xi32, 1>,
      %b: memref<?xi32, 1>,
      %c: memref<?xi1, 1>) {
    // CHECK-LABEL: llvm.func @less_dynamic_i32

    hip.less(%ctx) ins(%a, %b : memref<?xi32, 1>, memref<?xi32, 1>)
                   outs(%c : memref<?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_less({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
