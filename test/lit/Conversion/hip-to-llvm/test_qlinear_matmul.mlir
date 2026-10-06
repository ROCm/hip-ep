// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.qlinear_matmul -> wrap_qlinear_matmul
//
// UINT8 activations are HIPDNN_EP_DATATYPE_UINT8 = 7. INT8 weights are
// HIPDNN_EP_DATATYPE_INT8 = 5. Per-tensor scale and zero-point counts are 1.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @qlinear_matmul_row
  // CHECK-DAG: llvm.mlir.constant(4 : i64)
  // CHECK-DAG: llvm.mlir.constant(3 : i64)
  // CHECK-DAG: llvm.mlir.constant(7 : i64)
  // CHECK-DAG: llvm.mlir.constant(5 : i64)
  // CHECK: llvm.call @wrap_qlinear_matmul
  func.func @qlinear_matmul_row(
      %ctx: !hip.context,
      %a: memref<1x4xui8, 1>, %as: memref<f32, 1>, %az: memref<ui8, 1>,
      %b: memref<4x3xi8, 1>, %bs: memref<f32, 1>, %bz: memref<i8, 1>,
      %ys: memref<f32, 1>, %yz: memref<ui8, 1>,
      %y: memref<1x3xui8, 1>) {
    hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
        memref<1x4xui8, 1>, memref<f32, 1>, memref<ui8, 1>,
        memref<4x3xi8, 1>, memref<f32, 1>, memref<i8, 1>,
        memref<f32, 1>, memref<ui8, 1>)
      outs(%y : memref<1x3xui8, 1>)
    return
  }
}
