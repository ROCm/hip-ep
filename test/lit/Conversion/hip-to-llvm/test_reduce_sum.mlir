// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_sum operation is correctly lowered to LLVM call
// to wrap_reduce_sum runtime function with both static and dynamic shapes.
//
// This test validates:
// - hip.reduce_sum → llvm.call @wrap_reduce_sum
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: num_elements computed at compile time
// - Dynamic shapes: num_elements computed at runtime via extractvalue
// - Axes passed as pointer to runtime
//
// Expected: wrap_reduce_sum(state, data, axes, output, data_num_elements,
//                            output_num_elements, axes_num_elements,
//                            data_type, keepdims, noop_with_empty_axes)
// data_type is a HIPDNN_EP_DATATYPE_* enum value.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 3D tensor, reduce last axis
  func.func @reduce_sum_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_static_last_axis

    hip.reduce_sum(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // CHECK: llvm.call @wrap_reduce_sum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: Dynamic shapes
  func.func @reduce_sum_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_dynamic

    hip.reduce_sum(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}

    // Verify dynamic shape computation for data_num_elements
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64

    // CHECK: llvm.call @wrap_reduce_sum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: NON-trailing (strided) reduce — reduce the channel axis of an
  // NCHW-style tensor (the LayerNorm2d case). Shapes are chosen so inner_size
  // (=15) is distinct from output_num_elements (=4*1*3*5=60) and
  // data_num_elements (=4*8*3*5=480), so the CHECK below pins the inner_size
  // arg specifically. Trailing-match: [4,8,3,5] vs [4,1,3,5] -> dims 3,5 match,
  // axis 1 (8 vs 1) is the reduced axis -> inner = 3*5 = 15.
  func.func @reduce_sum_strided_channel_axis(
      %ctx: !hip.context,
      %input: memref<4x8x3x5xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<4x1x3x5xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_strided_channel_axis

    hip.reduce_sum(%ctx) ins(%input, %axes : memref<4x8x3x5xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<4x1x3x5xf32, 1>)
                         {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}

    // inner_size = 15 is passed as the trailing i64 arg.
    // CHECK-DAG: llvm.mlir.constant(15 : i64) : i64
    // CHECK: llvm.call @wrap_reduce_sum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
