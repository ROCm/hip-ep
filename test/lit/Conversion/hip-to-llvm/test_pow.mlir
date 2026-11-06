// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @pow_static(
      %ctx: !hip.context,
      %input: memref<4x8xf32, 1>,
      %output: memref<4x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @pow_static
    // CHECK: llvm.mlir.constant(2.200000e+00 : f64) : f64
    // CHECK: llvm.call @wrap_pow({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64) -> i32
    hip.pow(%ctx) ins(%input : memref<4x8xf32, 1>)
                  outs(%output : memref<4x8xf32, 1>)
                  {exponent = 2.2 : f64}
    return
  }

  func.func @pow_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @pow_dynamic_f16
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_pow({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64) -> i32
    hip.pow(%ctx) ins(%input : memref<?x?xf16, 1>)
                  outs(%output : memref<?x?xf16, 1>)
                  {exponent = 0.25 : f64}
    return
  }
}
