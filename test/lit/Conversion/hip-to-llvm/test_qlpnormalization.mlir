// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @qlpnormalization_u16(
      %ctx: !hip.context,
      %x: memref<1x8x16xui16, 1>,
      %y: memref<1x8x16xui16, 1>) {
    // CHECK-LABEL: llvm.func @qlpnormalization_u16
    // N = 16, UINT16 = 9, zp 3 / 7, p = 2, axis = -1
    // CHECK-DAG: llvm.mlir.constant(16 : i64)
    // CHECK-DAG: llvm.mlir.constant(9 : i64)
    // CHECK-DAG: llvm.mlir.constant(3 : i64)
    // CHECK-DAG: llvm.mlir.constant(7 : i64)
    // CHECK-DAG: llvm.mlir.constant(-1 : i64)
    // CHECK-DAG: llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @wrap_qlpnormalization({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64, f32, i64, i64, i64) -> i32
    hip.qlpnormalization(%ctx) ins(%x : memref<1x8x16xui16, 1>)
                               outs(%y : memref<1x8x16xui16, 1>)
                               {axis = -1 : i64, input_scale = 1.000000e-01 : f32,
                                input_zp = 3 : i64, output_scale = 2.000000e-01 : f32,
                                output_zp = 7 : i64, p = 2 : i64}
    return
  }
}
