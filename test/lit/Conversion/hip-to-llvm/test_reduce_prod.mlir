// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.reduce_prod lowers to llvm.call @wrap_reduce_prod sharing the
// same calling convention as reduce_sum / reduce_max:
//   (state, data, axes, output, data_num_elem, output_num_elem,
//    axes_num_elem, data_type, keepdims, noop_with_empty_axes,
//    inner_size) -> i32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  memref.global "private" constant @axis1 : memref<1xi64> = dense<[1]>

  // Static: reduce middle axis with keepdims=1.
  func.func @reduce_prod_static(
      %ctx: !hip.context,
      %data: memref<3x2x2xf32, 1>,
      %output: memref<3x1x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_prod_static

    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_prod(%ctx) ins(%data, %axes : memref<3x2x2xf32, 1>, memref<1xi64>)
                          outs(%output : memref<3x1x2xf32, 1>)
                          {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
                           normalized_axes = array<i64: 1>}

    // CHECK: llvm.call @wrap_reduce_prod({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Dynamic input shape.
  func.func @reduce_prod_dynamic(
      %ctx: !hip.context,
      %data: memref<?x2x?xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_prod_dynamic

    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_prod(%ctx) ins(%data, %axes : memref<?x2x?xf32, 1>, memref<1xi64>)
                          outs(%output : memref<?x?xf32, 1>)
                          {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
                           normalized_axes = array<i64: 1>}

    // CHECK: llvm.call @wrap_reduce_prod({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
    return
  }
}
