// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid_zero_extent(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x2x1x1xf32, 1>,
    %weights: memref<4x2x2x2xf32, 1>,
    %output: memref<1x4x0x0xf32, 1>) {
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x2x1x1xf32, 1>,
                           memref<4x2x2x2xf32, 1>)
    outs(%output : memref<1x4x0x0xf32, 1>)
    {kernel_shape = [2, 2], strides = [2, 2],
     pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
  return
}

// -----

func.func @wrong_spatial(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x2x8x8xf32, 1>,
    %weights: memref<4x2x3x3xf32, 1>,
    %output: memref<1x4x8x6xf32, 1>) {
  // expected-error @+1 {{dim 3 of result mismatch: expected 4}}
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x2x8x8xf32, 1>,
                           memref<4x2x3x3xf32, 1>)
    outs(%output : memref<1x4x8x6xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 2],
     pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
  return
}

// -----

func.func @wrong_group_channels(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x6x8x8xf32, 1>,
    %weights: memref<8x2x3x3xf32, 1>,
    %output: memref<1x8x8x8xf32, 1>) {
  // expected-error @+1 {{input channels 6 do not match weights channels-per-group 2 times group 2}}
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x6x8x8xf32, 1>,
                           memref<8x2x3x3xf32, 1>)
    outs(%output : memref<1x8x8x8xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [1, 1, 1, 1], dilations = [1, 1], group = 2}
  return
}

// -----

func.func @wrong_kernel_shape(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x2x8x8xf32, 1>,
    %weights: memref<4x2x3x3xf32, 1>,
    %output: memref<1x4x8x8xf32, 1>) {
  // expected-error @+1 {{kernel_shape dimension 5 does not match weights spatial dimension 3 at axis 0}}
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x2x8x8xf32, 1>,
                           memref<4x2x3x3xf32, 1>)
    outs(%output : memref<1x4x8x8xf32, 1>)
    {kernel_shape = [5, 3], strides = [1, 1],
     pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
  return
}

// -----

func.func @wide_window_product(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x?xf32, 1>,
    %weights: memref<1x1x9223372036854775807xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x1x?xf32, 1>,
                           memref<1x1x9223372036854775807xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {kernel_shape = [9223372036854775807], strides = [1],
     pads = [0, 0], dilations = [9223372036854775807], group = 1}
  return
}

// -----

func.func @window_offset_i64_boundary(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x?xf32, 1>,
    %weights: memref<1x1x9223372036854775807xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  hip.conv(%ctx)

    ins(%input, %weights : memref<1x1x?xf32, 1>,
                           memref<1x1x9223372036854775807xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {kernel_shape = [9223372036854775807], strides = [1],
     pads = [9223372036854775807, 9223372036854775807],
     dilations = [1], group = 1}
  return
}

// -----

func.func @static_invalid_padded_kernel(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x1x1xf32, 1>,
    %weights: memref<1x1x5x5xf32, 1>,
    %output: memref<1x1x?x?xf32, 1>) {
  // expected-error @+1 {{conv inferred a negative output extent -1 at spatial axis 0}}
  hip.conv(%ctx)
    ins(%input, %weights : memref<1x1x1x1xf32, 1>,
                           memref<1x1x5x5xf32, 1>)
    outs(%output : memref<1x1x?x?xf32, 1>)
    {kernel_shape = [5, 5], strides = [1, 1],
     pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
  return
}
