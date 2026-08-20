// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_mean operation is correctly lowered to an LLVM call to the
// wrap_reduce_mean runtime function with both static and dynamic shapes.
//
// hip.reduce_mean shares the (data, axes, output, keepdims, noop_with_empty_axes)
// signature with hip.reduce_sum and lowers through the same generic
// ReduceOpLowering template, so it emits the same 11-arg call (the trailing
// i64 is inner_size for strided / non-trailing-axis reduces).
//
// Expected: wrap_reduce_mean(state, data, axes, output, data_num_elements,
//                            output_num_elements, axes_num_elements,
//                            data_type, keepdims, noop_with_empty_axes,
//                            inner_size)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  memref.global "private" constant @axis1 : memref<1xi64> = dense<[1]>
  memref.global "private" constant @axis2 : memref<1xi64> = dense<[2]>

  // Test 1: Static 3D tensor, reduce last axis.
  func.func @reduce_mean_static_last_axis(
      %ctx: !hip.context,
      %input: memref<8x128x512xf16, 1>,
      %output: memref<8x128xf16, 1>) {
    // CHECK-LABEL: llvm.func @reduce_mean_static_last_axis

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_mean(%ctx) ins(%input, %axes : memref<8x128x512xf16, 1>, memref<1xi64>)
                          outs(%output : memref<8x128xf16, 1>)
                          {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                           normalized_axes = array<i64: 2>}

    // CHECK: %[[STATUS:.*]] = llvm.call @wrap_reduce_mean({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]]) : (!llvm.ptr, i32) -> i32

    return
  }

  // Test 2: Dynamic shapes.
  func.func @reduce_mean_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @reduce_mean_dynamic

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_mean(%ctx) ins(%input, %axes : memref<?x?x512xf16, 1>, memref<1xi64>)
                          outs(%output : memref<?x?xf16, 1>)
                          {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                           normalized_axes = array<i64: 2>}

    // CHECK: llvm.call @wrap_reduce_mean({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: NON-trailing (strided) reduce — channel axis of an NCHW-style
  // tensor (the LayerNorm2d / channel-mean case). Trailing-match
  // [4,8,3,5] vs [4,1,3,5] -> axis 1 is reduced -> inner_size = 3*5 = 15.
  func.func @reduce_mean_strided_channel_axis(
      %ctx: !hip.context,
      %input: memref<4x8x3x5xf16, 1>,
      %output: memref<4x1x3x5xf16, 1>) {
    // CHECK-LABEL: llvm.func @reduce_mean_strided_channel_axis

    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_mean(%ctx) ins(%input, %axes : memref<4x8x3x5xf16, 1>, memref<1xi64>)
                          outs(%output : memref<4x1x3x5xf16, 1>)
                          {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
                           normalized_axes = array<i64: 1>}

    // inner_size = 15 is passed as the trailing i64 arg.
    // CHECK-DAG: llvm.mlir.constant(15 : i64) : i64
    // CHECK: llvm.call @wrap_reduce_mean({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
