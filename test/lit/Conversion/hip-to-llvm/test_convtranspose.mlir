// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.conv_transpose is correctly lowered to an LLVM call into the
// transpose-convolution runtime wrapper.
//
// This test validates:
// - hip.conv_transpose → llvm.call @wrap_conv_transpose
// - Type conversion: !hip.context → !llvm.ptr
// - Static and dynamic memref dimensions (dynamic dims extracted from the
//   memref descriptor via llvm.extractvalue [3, N])
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. Static deconv with bias (f32)
  // --------------------------------------------------------------------------
  func.func @convtranspose_static_f32(
      %ctx: !hip.context,
      %input: memref<1x1x3x3xf32, 1>,
      %weights: memref<1x2x3x3xf32, 1>,
      %bias: memref<2xf32, 1>,
      %output: memref<1x2x5x5xf32, 1>) {
    // CHECK-LABEL: llvm.func @convtranspose_static_f32
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr
    hip.conv_transpose(%ctx) ins(%input, %weights, %bias : memref<1x1x3x3xf32, 1>,
                                                           memref<1x2x3x3xf32, 1>,
                                                           memref<2xf32, 1>)
                   outs(%output : memref<1x2x5x5xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    output_padding = [0, 0], group = 1}
    // CHECK: llvm.call @wrap_conv_transpose
    return
  }

  // --------------------------------------------------------------------------
  // 2. Dynamic batch dim (f16) — dim extracted from memref descriptor
  // --------------------------------------------------------------------------
  func.func @convtranspose_dynamic_f16(
      %ctx: !hip.context,
      %input: memref<?x1x3x3xf16, 1>,
      %weights: memref<1x2x3x3xf16, 1>,
      %output: memref<?x2x5x5xf16, 1>) {
    // CHECK-LABEL: llvm.func @convtranspose_dynamic_f16
    hip.conv_transpose(%ctx) ins(%input, %weights : memref<?x1x3x3xf16, 1>,
                                                    memref<1x2x3x3xf16, 1>)
                   outs(%output : memref<?x2x5x5xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1],
                    output_padding = [0, 0], group = 1}
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.call @wrap_conv_transpose
    return
  }
}
