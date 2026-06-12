// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_max operation is correctly lowered to LLVM call
// to wrap_reduce_max runtime function with both static and dynamic shapes.
//
// Expected: wrap_reduce_max(state, data, axes, output, data_num_elements,
//                            output_num_elements, axes_num_elements,
//                            data_type, keepdims, noop_with_empty_axes)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 3D tensor, reduce last axis
  func.func @reduce_max_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_max_static_last_axis

    hip.reduce_max(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK: llvm.call @wrap_reduce_max({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: Dynamic shapes
  func.func @reduce_max_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_max_dynamic

    hip.reduce_max(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64

    // CHECK: llvm.call @wrap_reduce_max({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
