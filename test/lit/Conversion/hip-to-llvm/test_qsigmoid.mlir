// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// hip.qsigmoid -> wrap_qsigmoid
//
// UINT16 storage must arrive as HIPDNN_EP_DATATYPE_UINT16 = 9. Input and
// output scales stay separate arguments because requant is required.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s
// ============================================================================

module {

// CHECK-LABEL: llvm.func @qsigmoid_ui16
// CHECK-DAG:   llvm.mlir.constant(128 : i64)
// CHECK-DAG:   llvm.mlir.constant(2048 : i64)
// CHECK-DAG:   llvm.mlir.constant(9 : i64)
// CHECK-DAG:   llvm.mlir.constant(35189 : i64)
// CHECK-DAG:   llvm.mlir.constant(35274 : i64)
// CHECK:       llvm.call @wrap_qsigmoid({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f32, i64, f32, i64) -> i32
  func.func @qsigmoid_ui16(%ctx: !hip.context,
                           %in: memref<1x128x2048xui16, 1>,
                           %out: memref<1x128x2048xui16, 1>) {
    hip.qsigmoid(%ctx) ins(%in : memref<1x128x2048xui16, 1>)
                       outs(%out : memref<1x128x2048xui16, 1>)
                       {input_scale = 1.000000e-01 : f32, input_zp = 35189 : i64,
                        output_scale = 2.000000e-01 : f32, output_zp = 35274 : i64}
    return
  }
}
