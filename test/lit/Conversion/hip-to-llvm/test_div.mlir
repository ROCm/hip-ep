// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.div lowers to llvm.call @wrap_div with the 4D-broadcast signature
//   (state, lhs, rhs, output,
//    lhs_n, lhs_c, lhs_h, lhs_w, rhs_n, rhs_c, rhs_h, rhs_w,
//    out_n, out_c, out_h, out_w, data_type) -> i32.
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

  func.func @div_dynamic_2d_f16(
      %ctx: !hip.context,
      %a: memref<?x?xf16, 1>,
      %b: memref<?x?xf16, 1>,
      %c: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @div_dynamic_2d_f16

    hip.div(%ctx) ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
                  outs(%c : memref<?x?xf16, 1>)

    // CHECK: llvm.call @wrap_div({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
