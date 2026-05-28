// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.div lowers to llvm.call @wrap_div with 4D shape-passing for
// ONNX broadcast (same pattern as hip.min / hip.add). Both same-shape and
// broadcasting cases are covered.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @div_static_2d_f32(
      %ctx: !hip.context,
      %a: memref<128x64xf32, 1>,
      %b: memref<128x64xf32, 1>,
      %c: memref<128x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @div_static_2d_f32

    hip.div(%ctx) ins(%a, %b : memref<128x64xf32, 1>, memref<128x64xf32, 1>)
                  outs(%c : memref<128x64xf32, 1>)

    // CHECK: llvm.call @wrap_div({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Broadcasting: rhs is per-channel [32] padded to [1,1,1,32].
  func.func @div_broadcast_f16(
      %ctx: !hip.context,
      %a: memref<1x128x32xf16, 1>,
      %b: memref<1x1x32xf16, 1>,
      %c: memref<1x128x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @div_broadcast_f16

    hip.div(%ctx) ins(%a, %b : memref<1x128x32xf16, 1>, memref<1x1x32xf16, 1>)
                  outs(%c : memref<1x128x32xf16, 1>)

    // CHECK: llvm.call @wrap_div({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  func.func @div_dynamic_2d_f16(
      %ctx: !hip.context,
      %a: memref<?x?xf16, 1>,
      %b: memref<?x?xf16, 1>,
      %c: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @div_dynamic_2d_f16

    hip.div(%ctx) ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
                  outs(%c : memref<?x?xf16, 1>)

    // CHECK: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: llvm.call @wrap_div({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
