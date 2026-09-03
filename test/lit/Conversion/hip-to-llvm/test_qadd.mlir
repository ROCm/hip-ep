// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @qadd_i8(
      %ctx: !hip.context,
      %lhs: memref<1x128x32xi8, 1>,
      %rhs: memref<1x128x32xi8, 1>,
      %out: memref<1x128x32xi8, 1>) {
    // CHECK-LABEL: llvm.func @qadd_i8

    hip.qadd(%ctx) ins(%lhs, %rhs : memref<1x128x32xi8, 1>, memref<1x128x32xi8, 1>)
                   outs(%out : memref<1x128x32xi8, 1>)
                   {lhs_scale = 1.000000e-01 : f32, lhs_zp = -5 : i64,
                    rhs_scale = 5.000000e-02 : f32, rhs_zp = 3 : i64,
                    output_scale = 2.000000e-01 : f32, output_zp = 7 : i64}

    // CHECK: llvm.call @wrap_qelementwise({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, f32, i64, f32, i64, i64) -> i32
    return
  }
}
