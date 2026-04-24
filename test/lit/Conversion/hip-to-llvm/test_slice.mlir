// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP slice operation is correctly lowered to an LLVM call to the
// wrap_slice runtime function.
//
// This test validates:
// - hip.slice -> llvm.call @wrap_slice
// - Type conversion: !hip.context -> !llvm.ptr
// - Optional inputs (axes, steps) passed as pointers (null when absent)
// - Stack-allocated i64 arrays for input/output shape metadata
// - Dynamic dim extraction via llvm.extractvalue for dynamic tensors
//
// Expected call signature:
//   wrap_slice(state, data, starts, ends, axes, steps, output,
//              input_shape, output_shape, rank, num_slice_entries,
//              output_num_elements, element_size_bytes, data_type)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: 3-operand slice (axes/steps absent).
  func.func @test_slice_basic(
      %ctx: !hip.context,
      %data: memref<4x8xf32, 1>,
      %starts: memref<2xi64, 1>,
      %ends: memref<2xi64, 1>,
      %output: memref<2x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @test_slice_basic
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<4x8xf32, 1>, memref<2xi64, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xf32, 1>)

    // Stack buffers for input/output shape metadata.
    // CHECK: llvm.alloca {{.*}} x i64
    // CHECK: llvm.alloca {{.*}} x i64
    // CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: 4-operand slice (axes provided, steps absent).
  func.func @test_slice_with_axes(
      %ctx: !hip.context,
      %data: memref<4x8xf32, 1>,
      %starts: memref<1xi64, 1>,
      %ends: memref<1xi64, 1>,
      %axes: memref<1xi64, 1>,
      %output: memref<4x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @test_slice_with_axes
    hip.slice(%ctx)
        ins(%data, %starts, %ends, %axes :
            memref<4x8xf32, 1>, memref<1xi64, 1>, memref<1xi64, 1>,
            memref<1xi64, 1>)
        outs(%output : memref<4x4xf32, 1>)

    // CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: 5-operand slice (axes and steps provided).
  func.func @test_slice_with_axes_and_steps(
      %ctx: !hip.context,
      %data: memref<4x8xf32, 1>,
      %starts: memref<2xi64, 1>,
      %ends: memref<2xi64, 1>,
      %axes: memref<2xi64, 1>,
      %steps: memref<2xi64, 1>,
      %output: memref<2x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @test_slice_with_axes_and_steps
    hip.slice(%ctx)
        ins(%data, %starts, %ends, %axes, %steps :
            memref<4x8xf32, 1>, memref<2xi64, 1>, memref<2xi64, 1>,
            memref<2xi64, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xf32, 1>)

    // CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 4: Dynamic input shape. The data dims come from the memref descriptor.
  func.func @test_slice_dynamic(
      %ctx: !hip.context,
      %data: memref<?x?xf32, 1>,
      %starts: memref<2xi64, 1>,
      %ends: memref<2xi64, 1>,
      %output: memref<2x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @test_slice_dynamic
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<?x?xf32, 1>, memref<2xi64, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xf32, 1>)

    // Dynamic dims extracted from the input descriptor (indices [3, 0..1]).
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

    return
  }
}
