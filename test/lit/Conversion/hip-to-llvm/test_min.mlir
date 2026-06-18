// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.min lowers to llvm.call @wrap_miopenOpTensor with the same
// 18-param shape-passing signature as hip.add / hip.mul, but with
// tensor_op = kTensorOpMin. Both same-shape and broadcasting cases are
// exercised so the per-operand 4D shape extraction is covered.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Same-shape 2D f32, both operands padded to [1,1,128,512].
  func.func @min_static_f32(
      %ctx: !hip.context,
      %a: memref<128x512xf32, 1>,
      %b: memref<128x512xf32, 1>,
      %c: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @min_static_f32

    hip.min(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%c : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
    return
  }

  // Broadcasting: rhs is a per-channel scalar [32] padded to [1,1,1,32].
  func.func @min_broadcast_f16(
      %ctx: !hip.context,
      %a: memref<1x128x32xf16, 1>,
      %b: memref<1x1x32xf16, 1>,
      %c: memref<1x128x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @min_broadcast_f16

    hip.min(%ctx) ins(%a, %b : memref<1x128x32xf16, 1>, memref<1x1x32xf16, 1>)
                  outs(%c : memref<1x128x32xf16, 1>)

    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> i32
    return
  }
}
