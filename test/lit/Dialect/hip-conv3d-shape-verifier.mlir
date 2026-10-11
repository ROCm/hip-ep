// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @conv3d_static(
    %ctx: !hip.context,
    %input: memref<2x4x9x10x11xf32, 1>,
    %weights: memref<8x2x3x5x2xf32, 1>,
    %output: memref<2x8x4x3x6xf32, 1>) {
  hip.conv(%ctx)
    ins(%input, %weights : memref<2x4x9x10x11xf32, 1>, memref<8x2x3x5x2xf32, 1>)
    outs(%output : memref<2x8x4x3x6xf32, 1>)
    {kernel_shape = [3, 5, 2], strides = [2, 3, 2],
     pads = [1, 2, 0, 0, 1, 2], dilations = [1, 1, 2], group = 2}
  return
}

// -----

func.func @conv3d_wrong_last_spatial(
    %ctx: !hip.context,
    %input: memref<2x4x9x10x11xf32, 1>,
    %weights: memref<8x2x3x5x2xf32, 1>,
    %output: memref<2x8x4x3x5xf32, 1>) {
  // expected-error @+1 {{dim 4 of result mismatch: expected 6}}
  hip.conv(%ctx)
    ins(%input, %weights : memref<2x4x9x10x11xf32, 1>, memref<8x2x3x5x2xf32, 1>)
    outs(%output : memref<2x8x4x3x5xf32, 1>)
    {kernel_shape = [3, 5, 2], strides = [2, 3, 2],
     pads = [1, 2, 0, 0, 1, 2], dilations = [1, 1, 2], group = 2}
  return
}

// -----

func.func @conv3d_rank_mismatch(
    %ctx: !hip.context,
    %input: memref<2x4x9x10x11xf32, 1>,
    %weights: memref<8x2x3x5xf32, 1>,
    %output: memref<2x8x4x3x6xf32, 1>) {
  // expected-error @+1 {{conv input and weights must have matching rank in [3, 5]}}
  hip.conv(%ctx)
    ins(%input, %weights : memref<2x4x9x10x11xf32, 1>, memref<8x2x3x5xf32, 1>)
    outs(%output : memref<2x8x4x3x6xf32, 1>)
    {kernel_shape = [3, 5, 2], strides = [2, 3, 2],
     pads = [1, 2, 0, 0, 1, 2], dilations = [1, 1, 2], group = 2}
  return
}
