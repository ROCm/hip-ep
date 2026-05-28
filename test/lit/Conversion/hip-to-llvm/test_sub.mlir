// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.sub lowers to llvm.call @wrap_elementwise_sub with 4D shape-
// passing for ONNX broadcast (same pattern as hip.div). Both same-shape and
// broadcasting cases are covered.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @sub_static_test(
      %ctx: !hip.context,
      %lhs: memref<128x512xf32, 1>,
      %rhs: memref<128x512xf32, 1>,
      %output: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_static_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%output : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Broadcasting: lhs scalar [1] padded to [1,1,1,1] over rhs [32].
  func.func @sub_broadcast_i64(
      %ctx: !hip.context,
      %lhs: memref<1xi64, 1>,
      %rhs: memref<32xi64, 1>,
      %output: memref<32xi64, 1>) {
    // CHECK-LABEL: llvm.func @sub_broadcast_i64

    hip.sub(%ctx) ins(%lhs, %rhs : memref<1xi64, 1>, memref<32xi64, 1>)
                  outs(%output : memref<32xi64, 1>)

    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  func.func @sub_dynamic_test(
      %ctx: !hip.context,
      %lhs: memref<?x512xf32, 1>,
      %rhs: memref<?x512xf32, 1>,
      %output: memref<?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_dynamic_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<?x512xf32, 1>, memref<?x512xf32, 1>)
                  outs(%output : memref<?x512xf32, 1>)

    // CHECK: llvm.mul {{.*}} : i64
    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
