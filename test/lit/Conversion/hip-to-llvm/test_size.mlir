// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.size lowers to a single llvm.call into wrap_size with the
// expected (state, output_ptr, num_elements) signature. The
// `num_elements` operand must be the product of all input dims, mixing
// compile-time constants (for static dims) and llvm.extractvalue over
// the MemRef descriptor's `sizes` array (for dynamic dims).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: fully dynamic 2D input. num_elements = sizes[0] * sizes[1].
  func.func @size_dynamic_2d(
      %ctx: !hip.context,
      %input: memref<?x?xf32, 1>,
      %output: memref<i64, 1>) {
    // CHECK-LABEL: llvm.func @size_dynamic_2d

    hip.size(%ctx) ins(%input : memref<?x?xf32, 1>)
                    outs(%output : memref<i64, 1>)

    // Dynamic dim chain: descriptor.sizes[0] * descriptor.sizes[1].
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64

    // CHECK: llvm.call @wrap_size({{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    return
  }

  // Test 2: mixed static/dynamic input — middle dim is `?`, outer/inner
  // dims are compile-time constants 2 and 4.
  func.func @size_dynamic_mixed(
      %ctx: !hip.context,
      %input: memref<2x?x4xf16, 1>,
      %output: memref<i64, 1>) {
    // CHECK-LABEL: llvm.func @size_dynamic_mixed

    hip.size(%ctx) ins(%input : memref<2x?x4xf16, 1>)
                    outs(%output : memref<i64, 1>)

    // Compile-time constants 2 and 4 must show up; the dynamic dim must
    // be loaded from the descriptor (sizes[1]).
    // CHECK-DAG: llvm.mlir.constant(2 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(4 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]

    // CHECK: llvm.call @wrap_size({{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    return
  }

  // Test 3: rank-1 dynamic input — num_elements = sizes[0] (still a
  // chain that starts at the unit constant).
  func.func @size_dynamic_1d(
      %ctx: !hip.context,
      %input: memref<?xi32, 1>,
      %output: memref<i64, 1>) {
    // CHECK-LABEL: llvm.func @size_dynamic_1d

    hip.size(%ctx) ins(%input : memref<?xi32, 1>)
                    outs(%output : memref<i64, 1>)

    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_size({{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    return
  }
}
