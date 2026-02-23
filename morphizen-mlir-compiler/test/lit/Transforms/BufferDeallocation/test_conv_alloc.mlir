// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify BufferDeallocation pass correctly handles temporary buffers used
// in convolution operations.
//
// This test validates:
// - Automatic deallocation of temporary buffers after convolution
// - No deallocation of function argument buffers (input, weights, output)
// - Proper placement of hip.free before function return
// - Deallocation after last use (after memref.copy)
//
// Pattern: Allocate temp → compute → copy to output → free temp
// This is common in conv+relu fusion where temp holds intermediate result.
//
// Expected: hip.free for temp buffer only, not for function arguments
// ============================================================================

// RUN: morphizen-opt %s --hip-buffer-deallocation | FileCheck %s

module {
  func.func @conv_test(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x3x3xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x224x224xf32, 1>) -> i32 {
    // CHECK-LABEL: func.func @conv_test

    // Allocate temporary buffer (owned by function)
    // CHECK: %[[TEMP:.*]] = hip.alloc(%{{.*}})
    %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>

    // Perform convolution into temp buffer
    hip.conv(%ctx, %input, %weights, %bias, %temp)
      {kernel_shape = [3, 3], strides = [1, 1],
       pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
      : (!hip.context, memref<1x3x224x224xf32, 1>,
         memref<64x3x3x3xf32, 1>, memref<64xf32, 1>, memref<1x64x224x224xf32, 1>)

    // Copy temp to output buffer (output is caller-owned)
    memref.copy %temp, %output : memref<1x64x224x224xf32, 1> to memref<1x64x224x224xf32, 1>

    %c0 = arith.constant 0 : i32
    // CHECK: arith.constant

    // BufferDeallocation should insert: hip.free(%ctx, %temp) before return
    // Should NOT free: input, weights, output (caller-owned)
    // CHECK-NEXT: hip.free(%{{.*}}, %[[TEMP]])
    // CHECK-NEXT: return
    return %c0 : i32
  }
}
