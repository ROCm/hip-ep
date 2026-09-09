// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @swish_static(
      %ctx: !hip.context,
      %input: memref<2x3xf32, 1>,
      %output: memref<2x3xf32, 1>) {
    // CHECK-LABEL: llvm.func @swish_static
    // CHECK: llvm.mlir.constant(5.000000e-01 : f64) : f64
    // CHECK: llvm.call @wrap_swish({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64) -> i32
    hip.swish(%ctx) ins(%input : memref<2x3xf32, 1>)
                    outs(%output : memref<2x3xf32, 1>)
                    {alpha = 0.5 : f64}
    return
  }

  func.func @swish_dynamic_bf16(
      %ctx: !hip.context,
      %input: memref<?x?x16xbf16, 1>,
      %output: memref<?x?x16xbf16, 1>) {
    // CHECK-LABEL: llvm.func @swish_dynamic_bf16
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(16 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_swish({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64) -> i32
    hip.swish(%ctx) ins(%input : memref<?x?x16xbf16, 1>)
                    outs(%output : memref<?x?x16xbf16, 1>)
    return
  }

  func.func @swish_f64(
      %ctx: !hip.context,
      %input: memref<8xf64, 1>,
      %output: memref<8xf64, 1>) {
    // CHECK-LABEL: llvm.func @swish_f64
    // CHECK-DAG: llvm.mlir.constant(6 : i64) : i64
    // CHECK: llvm.call @wrap_swish({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64) -> i32
    hip.swish(%ctx) ins(%input : memref<8xf64, 1>)
                    outs(%output : memref<8xf64, 1>)
    return
  }
}
