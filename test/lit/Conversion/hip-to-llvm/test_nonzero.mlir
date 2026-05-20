// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.nonzero lowers to a single llvm.call into the wrap_nonzero
// runtime stub, with the expected (state, in, out, num_elems, rank,
// output_capacity, input_data_type) signature. Covers both fully static
// inputs (num_elems computed at compile time) and partially dynamic
// inputs (num_elems built from llvm.extractvalue + llvm.mul over the
// MemRef descriptor sizes array).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 2D bool input -> [2, ?] i64 output.
  func.func @nonzero_static_bool(
      %ctx: !hip.context,
      %input: memref<3x4xi1, 1>,
      %output: memref<2x?xi64, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_static_bool

    hip.nonzero(%ctx) ins(%input : memref<3x4xi1, 1>)
                      outs(%output : memref<2x?xi64, 1>)
                      {input_data_type = 5 : i64}

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    return
  }

  // Test 2: Static 3D f32 input -> [3, ?] i64 output.
  func.func @nonzero_static_f32(
      %ctx: !hip.context,
      %input: memref<2x3x4xf32, 1>,
      %output: memref<3x?xi64, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_static_f32

    hip.nonzero(%ctx) ins(%input : memref<2x3x4xf32, 1>)
                      outs(%output : memref<3x?xi64, 1>)
                      {input_data_type = 0 : i64}

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    return
  }

  // Test 3: Dynamic input shape -> num_elements computed at runtime via
  // llvm.extractvalue + llvm.mul over the descriptor sizes array.
  func.func @nonzero_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %output: memref<2x?xi64, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_dynamic_f16

    hip.nonzero(%ctx) ins(%input : memref<?x?xf16, 1>)
                      outs(%output : memref<2x?xi64, 1>)
                      {input_data_type = 1 : i64}

    // Verify dynamic shape computation: descriptor.sizes[0] * descriptor.sizes[1].
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    return
  }
}
