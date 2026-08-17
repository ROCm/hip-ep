// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reduce_l2 operation is correctly lowered to an LLVM call to the
// wrap_reduce_l2 runtime function.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  memref.global "private" constant @axis2 : memref<1xi64> = dense<[2]>
  memref.global "private" constant @axis3 : memref<1xi64> = dense<[3]>

  func.func @reduce_l2_static_last_axis(
      %ctx: !hip.context,
      %input: memref<128x3x256x32xf16, 1>,
      %output: memref<128x3x256x1xf16, 1>) {
    // CHECK-LABEL: llvm.func @reduce_l2_static_last_axis

    %axes = memref.get_global @axis3 : memref<1xi64>
    hip.reduce_l2(%ctx) ins(%input, %axes : memref<128x3x256x32xf16, 1>, memref<1xi64>)
                        outs(%output : memref<128x3x256x1xf16, 1>)
                        {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
                         normalized_axes = array<i64: 3>}

    // CHECK: %[[STATUS:.*]] = llvm.call @wrap_reduce_l2({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]]) : (!llvm.ptr, i32) -> i32

    return
  }

  func.func @reduce_l2_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf16, 1>,
      %output: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @reduce_l2_dynamic

    %axes = memref.get_global @axis2 : memref<1xi64>
    hip.reduce_l2(%ctx) ins(%input, %axes : memref<?x?x512xf16, 1>, memref<1xi64>)
                        outs(%output : memref<?x?xf16, 1>)
                        {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                         normalized_axes = array<i64: 2>}

    // CHECK: llvm.call @wrap_reduce_l2({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
