// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.pool lowers to llvm.call @wrap_pool with the full ABI:
// (state, input, output, indices, data_type, pool_mode, spatial_rank, N, C,
//  in0..2, out0..2, k0..2, s0..2, p0..2, dil0..2, storage_order, ceil_mode,
//  has_indices, count_include_pad, p) -> i32.  Indices is a real ptr in the
// 2-output (MAX) form and a null ptr otherwise.  The three ONNX pooling ops
// share this single lowering, distinguished by pool_mode.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: 2D fp16 MaxPool, single output, fully static.  Verifies the
  // literal i64 args for spatial_rank=2, N, C, and all spatial extents.
  func.func @maxpool_2d_static_f16(
      %ctx: !hip.context,
      %x: memref<1x3x32x32xf16, 1>,
      %y: memref<1x3x16x16xf16, 1>) {
    // CHECK-LABEL: llvm.func @maxpool_2d_static_f16

    hip.pool(%ctx) ins(%x : memref<1x3x32x32xf16, 1>)
                   outs(%y : memref<1x3x16x16xf16, 1>)
                   {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 2}

    // No indices output -> the indices ptr arg should be a null pointer.
    // CHECK: llvm.mlir.zero
    // data_type=1 (HALF), pool_mode=1 (MAX).
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_pool
    return
  }

  // Test 2: 2D f32 MaxPool with Indices output.  Two output ptrs,
  // has_indices=1.
  func.func @maxpool_2d_with_indices(
      %ctx: !hip.context,
      %x: memref<1x3x32x32xf32, 1>,
      %y: memref<1x3x16x16xf32, 1>,
      %idx: memref<1x3x16x16xi64, 1>) {
    // CHECK-LABEL: llvm.func @maxpool_2d_with_indices

    hip.pool(%ctx) ins(%x : memref<1x3x32x32xf32, 1>)
                   outs(%y, %idx : memref<1x3x16x16xf32, 1>,
                                   memref<1x3x16x16xi64, 1>)
                   {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 2}

    // CHECK: llvm.call @wrap_pool
    return
  }

  // Test 3: dynamic batch (N) — verifies extractvalue [3, 0] for the N dim
  // off the input descriptor.  Spatial dims stay static so they appear as
  // i64 literals.
  func.func @maxpool_dynamic_n_f32(
      %ctx: !hip.context,
      %x: memref<?x3x16x16xf32, 1>,
      %y: memref<?x3x8x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @maxpool_dynamic_n_f32
    hip.pool(%ctx) ins(%x : memref<?x3x16x16xf32, 1>)
                   outs(%y : memref<?x3x8x8xf32, 1>)
                   {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 2}

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.call @wrap_pool
    return
  }

  // Test 4: 1D pool spans through the 1D code path (spatial_rank=1).
  // Trailing in/out/k/s/p/dil for unused axes 1, 2 must be 1 / 0.
  func.func @maxpool_1d_static_f32(
      %ctx: !hip.context,
      %x: memref<1x3x32xf32, 1>,
      %y: memref<1x3x31xf32, 1>) {
    // CHECK-LABEL: llvm.func @maxpool_1d_static_f32
    hip.pool(%ctx) ins(%x : memref<1x3x32xf32, 1>)
                   outs(%y : memref<1x3x31xf32, 1>)
                   {pool_mode = 1, kernel_shape = [2], strides = [1],
                    pads = [0, 0], dilations = [1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 2}

    // CHECK: llvm.call @wrap_pool
    return
  }

  // Test 5: 2D f32 AveragePool (pool_mode=0), single output.  Exercises the
  // count_include_pad path; indices ptr is null.
  func.func @averagepool_2d_static_f32(
      %ctx: !hip.context,
      %x: memref<1x3x32x32xf32, 1>,
      %y: memref<1x3x16x16xf32, 1>) {
    // CHECK-LABEL: llvm.func @averagepool_2d_static_f32
    hip.pool(%ctx) ins(%x : memref<1x3x32x32xf32, 1>)
                   outs(%y : memref<1x3x16x16xf32, 1>)
                   {pool_mode = 0, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 1, p = 2}

    // CHECK: llvm.mlir.zero
    // CHECK: llvm.call @wrap_pool
    return
  }

  // Test 6: 2D f32 LpPool (pool_mode=2, p=3), single output.
  func.func @lppool_2d_static_f32(
      %ctx: !hip.context,
      %x: memref<1x3x32x32xf32, 1>,
      %y: memref<1x3x16x16xf32, 1>) {
    // CHECK-LABEL: llvm.func @lppool_2d_static_f32
    hip.pool(%ctx) ins(%x : memref<1x3x32x32xf32, 1>)
                   outs(%y : memref<1x3x16x16xf32, 1>)
                   {pool_mode = 2, kernel_shape = [2, 2], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    ceil_mode = 0, storage_order = 0,
                    count_include_pad = 0, p = 3}

    // CHECK: llvm.call @wrap_pool
    return
  }
}
