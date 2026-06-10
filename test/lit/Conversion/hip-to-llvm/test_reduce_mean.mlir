// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @reduce_mean_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_mean_static_last_axis
    hip.reduce_mean(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64, 1>)
                          outs(%output : memref<8x128xf32, 1>)
                          {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    // CHECK: llvm.call @wrap_reduce_mean({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  func.func @reduce_mean_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_mean_dynamic
    hip.reduce_mean(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64, 1>)
                          outs(%output : memref<?x?xf32, 1>)
                          {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_reduce_mean
    return
  }
}
