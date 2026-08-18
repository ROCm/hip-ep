// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.expand lowers to llvm.call @wrap_expand_checked with signature
//   (state, in, shape, out, in_shape_ptr, in_rank,
//    out_shape_ptr, out_rank, shape_valid, data_type) -> i32, and that the
// wrapper status is recorded for ORT-visible failure propagation.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @checked_expand_extent(
      %ctx: !hip.context, %prior_valid: i1, %input: index, %target: index,
      %elements: index) -> (i1, index, index) {
    // CHECK-LABEL: llvm.func @checked_expand_extent
    // CHECK: %[[STATUS:.*]] = llvm.call @hipdnn_ep_checked_expand_extent
    // CHECK: %[[OK:.*]] = llvm.icmp "eq" %[[STATUS]]
    // CHECK: llvm.select %[[OK]]
    // CHECK: llvm.select %[[OK]]
    %valid, %extent, %next = hip.checked_expand_extent(
        %ctx, %prior_valid, %input, %target, %elements)
        expected_extent = -1
        -> (i1, index, index)
    return %valid, %extent, %next : i1, index, index
  }

  func.func @expand_static_f32(
      %ctx: !hip.context,
      %valid: i1,
      %x: memref<3x1xf32, 1>,
      %shape: memref<3xi64, 1>,
      %y: memref<2x3x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @expand_static_f32

    hip.expand(%ctx) valid(%valid) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
      outs(%y : memref<2x3x6xf32, 1>)

    // Input shape array (rank 2) + output shape array (rank 3).
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<3 x i64>
    // CHECK: %[[STATUS:.*]] = llvm.call @wrap_expand_checked({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
    // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[STATUS]])
    return
  }

  func.func @expand_dynamic(
      %ctx: !hip.context,
      %valid: i1,
      %x: memref<3x1xf32, 1>,
      %shape: memref<3xi64, 1>,
      %y: memref<?x3x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @expand_dynamic

    hip.expand(%ctx) valid(%valid) ins(%x, %shape : memref<3x1xf32, 1>, memref<3xi64, 1>)
      outs(%y : memref<?x3x?xf32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 2]
    // CHECK: %[[DYN_STATUS:.*]] = llvm.call @wrap_expand_checked({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
    // CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[DYN_STATUS]])
    return
  }
}
