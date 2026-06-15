// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.nonzero lowers to a single llvm.call into the wrap_nonzero
// runtime, with the expected 9-arg signature
//   (state, in, out, count_ptr, num_elems, rank, input_dims, capacity, dtype)
// where `count_ptr` is the device i32 scalar the kernel writes and
// `input_dims` is a host i64[R] stack array. Covers both fully static inputs
// (num_elems computed at compile time) and partially dynamic inputs
// (num_elems built from llvm.extractvalue + llvm.mul over the MemRef
// descriptor sizes array). Also verifies hip.readback_dim lowers to a single
// llvm.call into hipdnn_ep_readback_i32 followed by an sext to i64.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 2D bool input -> [2, ?] i64 output.
  func.func @nonzero_static_bool(
      %ctx: !hip.context,
      %input: memref<3x4xi1, 1>,
      %output: memref<2x?xi64, 1>,
      %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_static_bool

    hip.nonzero(%ctx) ins(%input : memref<3x4xi1, 1>)
                      outs(%output, %count : memref<2x?xi64, 1>, memref<i32, 1>)
                      {input_data_type = 5 : i64}

    // A host i64[R] dims array is stack-allocated and filled before the call.
    // CHECK: llvm.alloca {{.*}} x i64
    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Test 2: Static 3D f32 input -> [3, ?] i64 output.
  func.func @nonzero_static_f32(
      %ctx: !hip.context,
      %input: memref<2x3x4xf32, 1>,
      %output: memref<3x?xi64, 1>,
      %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_static_f32

    hip.nonzero(%ctx) ins(%input : memref<2x3x4xf32, 1>)
                      outs(%output, %count : memref<3x?xi64, 1>, memref<i32, 1>)
                      {input_data_type = 0 : i64}

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Test 3: Dynamic input shape -> num_elements computed at runtime via
  // llvm.extractvalue + llvm.mul over the descriptor sizes array.
  func.func @nonzero_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x?xf16, 1>,
      %output: memref<2x?xi64, 1>,
      %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nonzero_dynamic_f16

    hip.nonzero(%ctx) ins(%input : memref<?x?xf16, 1>)
                      outs(%output, %count : memref<2x?xi64, 1>, memref<i32, 1>)
                      {input_data_type = 1 : i64}

    // Verify dynamic shape computation: descriptor.sizes[0] * descriptor.sizes[1].
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]

    // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Test 4: hip.readback_dim lowers to hipdnn_ep_readback_i32 + zext to i64.
  func.func @readback_dim(
      %ctx: !hip.context,
      %count: memref<i32, 1>) -> index {
    // CHECK-LABEL: llvm.func @readback_dim
    %n = hip.readback_dim(%ctx, %count : memref<i32, 1>) -> index
    // CHECK: %[[C:.*]] = llvm.call @hipdnn_ep_readback_i32({{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK: llvm.zext %[[C]] : i32 to i64
    return %n : index
  }
}
