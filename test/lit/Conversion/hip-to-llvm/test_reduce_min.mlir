// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_min operation is correctly lowered to LLVM call
// to wrap_reduce_min runtime function with both static and dynamic shapes.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  memref.global "private" constant @axis2 : memref<1xi64> = dense<[2]>

  func.func @reduce_min_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_static_last_axis

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_min(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                          normalized_axes = array<i64: 2>}

    // CHECK: llvm.call @wrap_reduce_min({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  func.func @reduce_min_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_dynamic

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_min(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                          normalized_axes = array<i64: 2>}

    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64

    // CHECK: llvm.call @wrap_reduce_min({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
