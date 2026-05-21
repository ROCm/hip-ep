// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.nonzero lowers to a single llvm.call into the wrap_nonzero
// runtime stub, with the expected (state, input_ptr, input_num_elements,
// input_rank, input_data_type, slot_id, input_shape_host) signature.
// NonZero is strictly Category-C: the output sizes are unknown until the
// count kernel runs, so the runtime publishes the dynamic dim via
// `slot_id` and the EP reads it back post-compute. The `outs` operand is
// kept in IR for bufferization/aliasing but never written.
//
// Covers both fully static inputs (num_elems and shape array all constants)
// and partially dynamic inputs (num_elems / shape array slots built from
// llvm.extractvalue over the MemRef descriptor sizes[]).
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
                      {input_data_type = 5 : i64, slot_id = 0 : i32}

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32, !llvm.ptr) -> i32
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
                      {input_data_type = 0 : i64, slot_id = 1 : i32}

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32, !llvm.ptr) -> i32
    return
  }

  // Test 3: Dynamic input shape -> num_elements and shape array slots
  // both built at runtime via llvm.extractvalue over the descriptor's
  // sizes[] field.
  func.func @nonzero_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %output: memref<2x?xi64, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_dynamic_f16

    hip.nonzero(%ctx) ins(%input : memref<?x?xf16, 1>)
                      outs(%output : memref<2x?xi64, 1>)
                      {input_data_type = 1 : i64, slot_id = 2 : i32}

    // Verify dynamic shape computation: descriptor.sizes[0] * descriptor.sizes[1].
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32, !llvm.ptr) -> i32
    return
  }
}
