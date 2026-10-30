// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_min operation is correctly lowered to LLVM call
// to wrap_reduce_min runtime function with both static and dynamic shapes.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @reduce_min_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_static_last_axis

    hip.reduce_min(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK: llvm.call @wrap_reduce_min({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  func.func @reduce_min_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_dynamic

    hip.reduce_min(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64

    // CHECK: llvm.call @wrap_reduce_min({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // keepdims=0 deletes the last axis. [1,6,5,5] -> [1,6,5] matches that 5
  // from the trailing end, which used to pass inner_size=5. Axis 3 is the
  // last axis, so the inner suffix is empty.
  memref.global "private" constant @axes_last : memref<1xi64> = dense<3>
  func.func @reduce_min_repeated_last_dim(
      %ctx: !hip.context,
      %input: memref<1x6x5x5xf32, 1>,
      %output: memref<1x6x5xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_repeated_last_dim
    %axes = memref.get_global @axes_last : memref<1xi64>
    hip.reduce_min(%ctx) ins(%input, %axes : memref<1x6x5x5xf32, 1>, memref<1xi64>)
                         outs(%output : memref<1x6x5xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // dtype, keepdims, and noop are 0. The next constant is inner_size.
    // CHECK: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[INNER:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: llvm.call @wrap_reduce_min({{.*}}, %[[INNER]])
    return
  }

  // Same output shape as reducing axis 0 or axis 2. Trailing alignment of
  // [4,4,4] -> [4,4] consumes every 4 and would pass inner_size=16. Axis 1
  // leaves one 4 after the reduced dim.
  memref.global "private" constant @axes_middle : memref<1xi64> = dense<1>
  func.func @reduce_min_middle_of_repeated_dims(
      %ctx: !hip.context,
      %input: memref<4x4x4xf32, 1>,
      %output: memref<4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_min_middle_of_repeated_dims
    %axes = memref.get_global @axes_middle : memref<1xi64>
    hip.reduce_min(%ctx) ins(%input, %axes : memref<4x4x4xf32, 1>, memref<1xi64>)
                         outs(%output : memref<4x4xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[INNER:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK-NEXT: llvm.call @wrap_reduce_min({{.*}}, %[[INNER]])
    return
  }
}
