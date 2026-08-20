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
//                            data_type, keepdims, noop_with_empty_axes,
//                            inner_size)
// data_type is a HIPDNN_EP_DATATYPE_* enum value.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  memref.global "private" constant @axis0 : memref<1xi64> = dense<[0]>
  memref.global "private" constant @axis1 : memref<1xi64> = dense<[1]>
  memref.global "private" constant @axis2 : memref<1xi64> = dense<[2]>
  memref.global "private" constant @axes01 : memref<2xi64> = dense<[0, 1]>

  // Test 1: Static 3D tensor, reduce last axis
  func.func @reduce_sum_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf32, 1>,
      %output: memref<8x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_static_last_axis

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_sum(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                          normalized_axes = array<i64: 2>}

    // CHECK: %[[STATUS:.*]] = llvm.call @wrap_reduce_sum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]]) : (!llvm.ptr, i32) -> i32

    return
  }

  // Test 2: Dynamic shapes
  func.func @reduce_sum_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_dynamic

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_sum(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                          normalized_axes = array<i64: 2>}

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
      %output: memref<4x1x3x5xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_strided_channel_axis

    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_sum(%ctx) ins(%input, %axes : memref<4x8x3x5xf32, 1>, memref<1xi64>)
                         outs(%output : memref<4x1x3x5xf32, 1>)
                         {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
                          normalized_axes = array<i64: 1>}

    // inner_size = 15 is passed as the trailing i64 arg.
    // CHECK-DAG: llvm.mlir.constant(15 : i64) : i64
    // CHECK: llvm.call @wrap_reduce_sum({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Equal input extents make axis 0 and axis 1 indistinguishable from shapes
  // alone. The validated constant source must select inner_size=4 for axis 0.
  func.func @reduce_sum_equal_extents_axis0(
      %ctx: !hip.context,
      %input: memref<4x4xf32, 1>,
      %output: memref<4xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_equal_extents_axis0
    %axes = memref.get_global @axis0 : memref<1xi64>
    hip.reduce_sum(%ctx)
      ins(%input, %axes : memref<4x4xf32, 1>, memref<1xi64>)
      outs(%output : memref<4xf32, 1>)
      {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
       normalized_axes = array<i64: 0>}

    // CHECK: llvm.mlir.addressof @axis0
    // CHECK: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[INNER0:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK-NEXT: llvm.call @wrap_reduce_sum({{.*}}, %[[INNER0]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // The same shapes with axis 1 instead require inner_size=1.
  func.func @reduce_sum_equal_extents_axis1(
      %ctx: !hip.context,
      %input: memref<4x4xf32, 1>,
      %output: memref<4xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_equal_extents_axis1
    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_sum(%ctx)
      ins(%input, %axes : memref<4x4xf32, 1>, memref<1xi64>)
      outs(%output : memref<4xf32, 1>)
      {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
       normalized_axes = array<i64: 1>}

    // CHECK: llvm.mlir.addressof @axis1
    // CHECK: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[INNER1:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: llvm.call @wrap_reduce_sum({{.*}}, %[[INNER1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // A contiguous multi-axis span is flattened by the runtime. For axes [0, 1],
  // dimension 2 remains after the span, so inner_size=3.
  func.func @reduce_sum_contiguous_axis_span(
      %ctx: !hip.context,
      %input: memref<4x5x3xf32, 1>,
      %output: memref<3xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_contiguous_axis_span
    %axes = memref.get_global @axes01 : memref<2xi64>
    hip.reduce_sum(%ctx)
      ins(%input, %axes : memref<4x5x3xf32, 1>, memref<2xi64>)
      outs(%output : memref<3xf32, 1>)
      {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
       normalized_axes = array<i64: 0, 1>}

    // CHECK: llvm.mlir.addressof @axes01
    // CHECK: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[INNERSPAN:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK-NEXT: llvm.call @wrap_reduce_sum({{.*}}, %[[INNERSPAN]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
