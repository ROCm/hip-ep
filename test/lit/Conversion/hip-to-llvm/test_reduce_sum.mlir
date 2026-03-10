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
//                            output_num_elements, element_size_bytes, keepdims)
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
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.reduce_sum(%ctx) ins(%input, %axes : memref<8x128x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<8x128xf32, 1>)
                         {keepdims = 0 : i64}

    // CHECK: llvm.call @wrap_reduce_sum(%[[CTX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: Dynamic shapes
  func.func @reduce_sum_dynamic(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reduce_sum_dynamic
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.reduce_sum(%ctx) ins(%input, %axes : memref<?x?x512xf32, 1>, memref<1xi64, 1>)
                         outs(%output : memref<?x?xf32, 1>)
                         {keepdims = 0 : i64}

    // Verify dynamic shape computation for data_num_elements
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[DIM0:.*]] = llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: %[[PROD1:.*]] = llvm.mul %[[ONE]], %[[DIM0]] : i64
    // CHECK: %[[DIM1:.*]] = llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: %[[PROD2:.*]] = llvm.mul %[[PROD1]], %[[DIM1]] : i64
    // CHECK: %[[DIM2:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[DATA_NUM_ELEMENTS:.*]] = llvm.mul %[[PROD2]], %[[DIM2]] : i64

    // CHECK: llvm.call @wrap_reduce_sum(%[[CTX]], %{{.*}}, %{{.*}}, %{{.*}}, %[[DATA_NUM_ELEMENTS]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }
}
