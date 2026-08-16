// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid(%ctx: !hip.context,
                 %input: memref<1x1x3x3xf32, 1>,
                 %weights: memref<1x2x3x3xf32, 1>,
                 %bias: memref<2xf32, 1>,
                 %output: memref<1x2x5x5xf32, 1>) {
  hip.conv_transpose(%ctx)
    ins(%input, %weights, %bias : memref<1x1x3x3xf32, 1>,
                                  memref<1x2x3x3xf32, 1>,
                                  memref<2xf32, 1>)
    outs(%output : memref<1x2x5x5xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @wrong_output_channels(
    %ctx: !hip.context,
    %input: memref<1x1x3x3xf32, 1>,
    %weights: memref<1x2x3x3xf32, 1>,
    %output: memref<1x4x5x5xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 2}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x1x3x3xf32, 1>,
                             memref<1x2x3x3xf32, 1>)
    outs(%output : memref<1x4x5x5xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @wrong_output_spatial(
    %ctx: !hip.context,
    %input: memref<1x1x3x3xf32, 1>,
    %weights: memref<1x2x3x3xf32, 1>,
    %output: memref<1x2x6x5xf32, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 5}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x1x3x3xf32, 1>,
                             memref<1x2x3x3xf32, 1>)
    outs(%output : memref<1x2x6x5xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @wrong_input_channels(
    %ctx: !hip.context,
    %input: memref<1x3x3x3xf32, 1>,
    %weights: memref<1x2x3x3xf32, 1>,
    %output: memref<1x2x5x5xf32, 1>) {
  // expected-error @+1 {{input channels 3 do not match weights dimension 0 1}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x3x3x3xf32, 1>,
                             memref<1x2x3x3xf32, 1>)
    outs(%output : memref<1x2x5x5xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @wrong_bias(
    %ctx: !hip.context,
    %input: memref<1x1x3x3xf32, 1>,
    %weights: memref<1x2x3x3xf32, 1>,
    %bias: memref<3xf32, 1>,
    %output: memref<1x2x5x5xf32, 1>) {
  // expected-error @+1 {{bias length must match output channels}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights, %bias : memref<1x1x3x3xf32, 1>,
                                  memref<1x2x3x3xf32, 1>,
                                  memref<3xf32, 1>)
    outs(%output : memref<1x2x5x5xf32, 1>)
    {kernel_shape = [3, 3], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @output_channel_product_overflow(
    %ctx: !hip.context,
    %input: memref<1x1x1x1xf32, 1>,
    %weights: memref<1x9223372036854775807x1x1xf32, 1>,
    %output: memref<1x?x1x1xf32, 1>) {
  // expected-error @+1 {{conv_transpose output channels inferred an out-of-range output extent}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x1x1x1xf32, 1>,
                             memref<1x9223372036854775807x1x1xf32, 1>)
    outs(%output : memref<1x?x1x1xf32, 1>)
    {kernel_shape = [1, 1], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 2}
  return
}

// -----

func.func @spatial_offset_overflow(
    %ctx: !hip.context,
    %input: memref<1x1x?x1xf32, 1>,
    %weights: memref<1x1x1x1xf32, 1>,
    %output: memref<1x1x?x1xf32, 1>) {
  // expected-error @+1 {{conv_transpose affine offset is out of range at spatial axis 0}}
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x1x?x1xf32, 1>,
                             memref<1x1x1x1xf32, 1>)
    outs(%output : memref<1x1x?x1xf32, 1>)
    {kernel_shape = [9223372036854775807, 1], strides = [1, 1],
     pads = [0, 0, 0, 0],
     dilations = [9223372036854775807, 1],
     output_padding = [0, 0], group = 1}
  return
}

// -----

func.func @channel_and_offset_i64_boundary(
    %ctx: !hip.context,
    %input: memref<1x1x?x1xf32, 1>,
    %weights: memref<1x9223372036854775807x1x1xf32, 1>,
    %output: memref<1x9223372036854775807x?x1xf32, 1>) {
  hip.conv_transpose(%ctx)
    ins(%input, %weights : memref<1x1x?x1xf32, 1>,
                             memref<1x9223372036854775807x1x1xf32, 1>)
    outs(%output : memref<1x9223372036854775807x?x1xf32, 1>)
    {kernel_shape = [9223372036854775807, 1], strides = [1, 1],
     pads = [0, 0, 0, 0], dilations = [1, 1],
     output_padding = [0, 0], group = 1}
  return
}
