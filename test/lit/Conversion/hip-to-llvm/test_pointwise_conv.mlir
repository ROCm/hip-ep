// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the hip.conv HIP->LLVM lowering routes a 1x1 / stride-1 / no-pad /
// no-dilation / group-1 conv with small Cin and static shapes to the fused
// GEMM+bias custom kernel (wrap_pointwise_conv), and that everything outside
// that envelope falls back to wrap_miopenConvolutionForward.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Small-Cin (16) static 1x1 conv -> fused pointwise kernel.
  // CHECK-LABEL: llvm.func @pointwise_small_cin
  // CHECK: llvm.call @wrap_pointwise_conv
  // CHECK-NOT: llvm.call @wrap_miopenConvolutionForward
  func.func @pointwise_small_cin(
      %ctx: !hip.context,
      %input: memref<1x16x64x64xf16, 1>,
      %weights: memref<256x16x1x1xf16, 1>,
      %bias: memref<256xf16, 1>,
      %output: memref<1x256x64x64xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x16x64x64xf16, 1>,
                                                 memref<256x16x1x1xf16, 1>,
                                                 memref<256xf16, 1>)
                   outs(%output : memref<1x256x64x64xf16, 1>)
                   {kernel_shape = [1, 1], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // Large-Cin (128 > threshold) 1x1 conv -> MIOpen fallback.
  // CHECK-LABEL: llvm.func @pointwise_large_cin
  // CHECK: llvm.call @wrap_miopenConvolutionForward
  // CHECK-NOT: llvm.call @wrap_pointwise_conv
  func.func @pointwise_large_cin(
      %ctx: !hip.context,
      %input: memref<1x128x64x64xf16, 1>,
      %weights: memref<256x128x1x1xf16, 1>,
      %bias: memref<256xf16, 1>,
      %output: memref<1x256x64x64xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x128x64x64xf16, 1>,
                                                 memref<256x128x1x1xf16, 1>,
                                                 memref<256xf16, 1>)
                   outs(%output : memref<1x256x64x64xf16, 1>)
                   {kernel_shape = [1, 1], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // 1x1 conv with stride 2 -> not pointwise -> MIOpen fallback.
  // CHECK-LABEL: llvm.func @pointwise_strided
  // CHECK: llvm.call @wrap_miopenConvolutionForward
  // CHECK-NOT: llvm.call @wrap_pointwise_conv
  func.func @pointwise_strided(
      %ctx: !hip.context,
      %input: memref<1x16x64x64xf16, 1>,
      %weights: memref<256x16x1x1xf16, 1>,
      %bias: memref<256xf16, 1>,
      %output: memref<1x256x32x32xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x16x64x64xf16, 1>,
                                                 memref<256x16x1x1xf16, 1>,
                                                 memref<256xf16, 1>)
                   outs(%output : memref<1x256x32x32xf16, 1>)
                   {kernel_shape = [1, 1], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // Dynamic batch 1x1 conv -> not static -> MIOpen fallback.
  // CHECK-LABEL: llvm.func @pointwise_dynamic
  // CHECK: llvm.call @wrap_miopenConvolutionForward
  // CHECK-NOT: llvm.call @wrap_pointwise_conv
  func.func @pointwise_dynamic(
      %ctx: !hip.context,
      %input: memref<?x16x64x64xf16, 1>,
      %weights: memref<256x16x1x1xf16, 1>,
      %bias: memref<256xf16, 1>,
      %output: memref<?x256x64x64xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<?x16x64x64xf16, 1>,
                                                 memref<256x16x1x1xf16, 1>,
                                                 memref<256xf16, 1>)
                   outs(%output : memref<?x256x64x64xf16, 1>)
                   {kernel_shape = [1, 1], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // 3x3 conv -> not 1x1 -> MIOpen fallback.
  // CHECK-LABEL: llvm.func @conv_3x3
  // CHECK: llvm.call @wrap_miopenConvolutionForward
  // CHECK-NOT: llvm.call @wrap_pointwise_conv
  func.func @conv_3x3(
      %ctx: !hip.context,
      %input: memref<1x16x64x64xf16, 1>,
      %weights: memref<256x16x3x3xf16, 1>,
      %bias: memref<256xf16, 1>,
      %output: memref<1x256x64x64xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x16x64x64xf16, 1>,
                                                 memref<256x16x3x3xf16, 1>,
                                                 memref<256xf16, 1>)
                   outs(%output : memref<1x256x64x64xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    return
  }
}
