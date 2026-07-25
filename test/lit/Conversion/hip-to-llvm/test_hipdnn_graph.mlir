// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.hipdnn_graph is correctly lowered to LLVM call to the
// hipdnn_graph_execute C runtime function with stack-allocated UID and
// pointer arrays (variant_pack dispatch pattern).
//
// This test validates:
// - hip.hipdnn_graph → llvm.call @hipdnn_graph_execute
// - graph_id and num_io are lowered as i32 constants
// - UIDs are stored into a stack-allocated i64 array
// - Memref aligned_ptrs are stored into a stack-allocated ptr array
// - 5-param signature: state_ptr, graph_id, num_io, uids_ptr, ptrs_ptr
//
// Expected: hipdnn_graph_execute(state, graph_id=0, num_io=3,
//             uids=[0,1,2], ptrs=[x_ptr, w_ptr, out_ptr])
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Conv-like 2-input 1-output graph (graph_id=0)
  func.func @test_hipdnn_graph_conv(
      %ctx: !hip.context,
      %x: memref<1x1x8x8xf32>,
      %w: memref<1x1x3x3xf32>,
      %out: memref<1x1x8x8xf32>) {
    // CHECK-LABEL: llvm.func @test_hipdnn_graph_conv

    hip.hipdnn_graph(%ctx) graph_id(0)
        ins(%x, %w : memref<1x1x8x8xf32>, memref<1x1x3x3xf32>)
        outs(%out : memref<1x1x8x8xf32>)
        {input_uids = [0, 1], output_uids = [2]}

    // graph_id = 0, num_io = 3 (2 inputs + 1 output)
    // CHECK-DAG:   %[[GID:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG:   %[[NIO:.*]] = llvm.mlir.constant(3 : i32) : i32

    // Stack arrays for UIDs (i64) and pointers
    // CHECK-DAG:   %[[UIDS:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x i64>
    // CHECK-DAG:   %[[PTRS:.*]] = llvm.alloca {{.*}} x !llvm.array<3 x ptr>

    // UID stores: input_uids=[0,1], output_uids=[2]
    // CHECK:       llvm.store
    // CHECK:       llvm.store
    // CHECK:       llvm.store

    // Pointer stores: x_ptr, w_ptr, out_ptr
    // CHECK:       llvm.store
    // CHECK:       llvm.store
    // CHECK:       llvm.store

    // CHECK:       llvm.call @hipdnn_graph_execute({{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

    return
  }

  // Test 2: Different graph_id (graph_id=5) to verify ID propagation
  func.func @test_hipdnn_graph_id(
      %ctx: !hip.context,
      %x: memref<4x3x224x224xf32>,
      %out: memref<4x3x224x224xf32>) {
    // CHECK-LABEL: llvm.func @test_hipdnn_graph_id

    hip.hipdnn_graph(%ctx) graph_id(5)
        ins(%x : memref<4x3x224x224xf32>)
        outs(%out : memref<4x3x224x224xf32>)
        {input_uids = [10], output_uids = [20]}

    // graph_id = 5, num_io = 2 (1 input + 1 output)
    // CHECK-DAG:   %[[GID2:.*]] = llvm.mlir.constant(5 : i32) : i32
    // CHECK-DAG:   %[[NIO2:.*]] = llvm.mlir.constant(2 : i32) : i32

    // Stack arrays for 2 entries
    // CHECK-DAG:   %[[UIDS2:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK-DAG:   %[[PTRS2:.*]] = llvm.alloca {{.*}} x !llvm.array<2 x ptr>

    // CHECK:       llvm.call @hipdnn_graph_execute({{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

    return
  }

  // Test 3: f16 element type to verify type-agnostic dispatch
  func.func @test_hipdnn_graph_f16(
      %ctx: !hip.context,
      %x: memref<1x64x56x56xf16>,
      %w: memref<64x64x3x3xf16>,
      %out: memref<1x64x56x56xf16>) {
    // CHECK-LABEL: llvm.func @test_hipdnn_graph_f16

    hip.hipdnn_graph(%ctx) graph_id(2)
        ins(%x, %w : memref<1x64x56x56xf16>, memref<64x64x3x3xf16>)
        outs(%out : memref<1x64x56x56xf16>)
        {input_uids = [100, 101], output_uids = [200]}

    // num_io = 3 regardless of element type
    // CHECK:       llvm.call @hipdnn_graph_execute({{.*}}) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

    return
  }
}
