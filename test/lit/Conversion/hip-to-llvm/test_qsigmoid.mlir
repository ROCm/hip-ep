// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK: llvm.func @wrap_qactivation(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64, f32, i64) -> i32

module {
  func.func @qsigmoid_i8(
      %ctx: !hip.context,
      %x: memref<1x128x32xi8, 1>,
      %y: memref<1x128x32xi8, 1>) {
    // CHECK-LABEL: llvm.func @qsigmoid_i8

    hip.qsigmoid(%ctx) ins(%x : memref<1x128x32xi8, 1>)
                       outs(%y : memref<1x128x32xi8, 1>)
                       {x_scale = 2.500000e-01 : f32, x_zero_point = -5 : i64,
                        y_scale = 1.250000e-01 : f32, y_zero_point = 4 : i64}

    // out_recip_scale folds y_scale's division into a compile-time constant:
    // out_recip_scale = 1 / y_scale = 1 / 0.125 = 8.0
    // CHECK-DAG: llvm.mlir.constant(8.000000e+00 : f32)
    // CHECK-DAG: llvm.mlir.constant(2.500000e-01 : f32)
    // CHECK: llvm.call @wrap_qactivation({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64, f32, i64) -> i32
    return
  }
}
