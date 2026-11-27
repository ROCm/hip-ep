// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @state_channels_last_without_channels_last(%ctx: !hip.context,
    %input: memref<1x64x8xf16, 1>, %weight: memref<64x1x4xf16, 1>,
    %output: memref<1x64x8xf16, 1>, %present: memref<1x3x64xf16, 1>) {
  // expected-error @+1 {{state_channels_last requires channels_last}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<1x64x8xf16, 1>, memref<64x1x4xf16, 1>)
      outs(%output, %present :
           memref<1x64x8xf16, 1>, memref<1x3x64xf16, 1>)
      {activation = "silu", ndim = 1 : i64, state_channels_last = true}
  return
}

// -----

func.func @dilation_without_state_channels_last(%ctx: !hip.context,
    %input: memref<1x8x64xf16, 1>, %weight: memref<64x1x4xf16, 1>,
    %output: memref<1x8x64xf16, 1>, %present: memref<1x64x6xf16, 1>) {
  // expected-error @+1 {{dilation > 1 requires state_channels_last}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<1x8x64xf16, 1>, memref<64x1x4xf16, 1>)
      outs(%output, %present :
           memref<1x8x64xf16, 1>, memref<1x64x6xf16, 1>)
      {activation = "silu", ndim = 1 : i64, channels_last = true,
       dilation = 2 : i64}
  return
}

// -----

func.func @state_len_mismatch(%ctx: !hip.context,
    %input: memref<1x8x64xf16, 1>, %weight: memref<64x1x4xf16, 1>,
    %past: memref<1x3x64xf16, 1>,
    %output: memref<1x8x64xf16, 1>, %present: memref<1x3x64xf16, 1>) {
  // expected-error @+1 {{state dim 1 must be (kernel_size - 1) * dilation = 6, got 3}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<1x8x64xf16, 1>, memref<64x1x4xf16, 1>)
      past_state(%past : memref<1x3x64xf16, 1>)
      outs(%output, %present :
           memref<1x8x64xf16, 1>, memref<1x3x64xf16, 1>)
      {activation = "silu", ndim = 1 : i64, channels_last = true,
       dilation = 2 : i64, state_channels_last = true}
  return
}

// -----

func.func @state_channels_mismatch(%ctx: !hip.context,
    %input: memref<1x8x64xf16, 1>, %weight: memref<64x1x4xf16, 1>,
    %output: memref<1x8x64xf16, 1>, %present: memref<1x3x32xf16, 1>) {
  // expected-error @+1 {{state dim 2 must match the input channels 64, got 32}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<1x8x64xf16, 1>, memref<64x1x4xf16, 1>)
      outs(%output, %present :
           memref<1x8x64xf16, 1>, memref<1x3x32xf16, 1>)
      {activation = "silu", ndim = 1 : i64, channels_last = true,
       state_channels_last = true}
  return
}

// -----

func.func @varlen_cumulative_length_mismatch(%ctx: !hip.context,
    %input: memref<6x4xf16, 1>, %weight: memref<4x1x4xf16, 1>,
    %cumulative: memref<4xi32, 1>, %state: memref<2x4x3xf16, 1>,
    %output: memref<6x4xf16, 1>, %final: memref<2x4x3xf16, 1>) {
  // expected-error @+1 {{cumulative_sequence_length must have batch + 1 entries}}
  hip.varlen_causal_conv_with_state(%ctx)
      ins(%input, %weight, %cumulative, %state :
          memref<6x4xf16, 1>, memref<4x1x4xf16, 1>, memref<4xi32, 1>,
          memref<2x4x3xf16, 1>)
      outs(%output, %final : memref<6x4xf16, 1>, memref<2x4x3xf16, 1>)
      {activation = "silu"}
  return
}

// -----

func.func @varlen_output_shape_mismatch(%ctx: !hip.context,
    %input: memref<6x4xf16, 1>, %weight: memref<4x1x4xf16, 1>,
    %cumulative: memref<3xi32, 1>, %state: memref<2x4x3xf16, 1>,
    %output: memref<5x4xf16, 1>, %final: memref<2x4x3xf16, 1>) {
  // expected-error @+1 {{output shape must match the input shape}}
  hip.varlen_causal_conv_with_state(%ctx)
      ins(%input, %weight, %cumulative, %state :
          memref<6x4xf16, 1>, memref<4x1x4xf16, 1>, memref<3xi32, 1>,
          memref<2x4x3xf16, 1>)
      outs(%output, %final : memref<5x4xf16, 1>, memref<2x4x3xf16, 1>)
      {activation = "silu"}
  return
}

// -----

func.func @varlen_element_type_mismatch(%ctx: !hip.context,
    %input: memref<6x4xf16, 1>, %weight: memref<4x1x4xf32, 1>,
    %cumulative: memref<3xi32, 1>, %state: memref<2x4x3xf16, 1>,
    %output: memref<6x4xf16, 1>, %final: memref<2x4x3xf16, 1>) {
  // expected-error @+1 {{weight element type must match the input element type}}
  hip.varlen_causal_conv_with_state(%ctx)
      ins(%input, %weight, %cumulative, %state :
          memref<6x4xf16, 1>, memref<4x1x4xf32, 1>, memref<3xi32, 1>,
          memref<2x4x3xf16, 1>)
      outs(%output, %final : memref<6x4xf16, 1>, memref<2x4x3xf16, 1>)
      {activation = "silu"}
  return
}

// -----

func.func @varlen_weight_channels_mismatch(%ctx: !hip.context,
    %input: memref<6x4xf16, 1>, %weight: memref<5x1x4xf16, 1>,
    %cumulative: memref<3xi32, 1>, %state: memref<2x4x3xf16, 1>,
    %output: memref<6x4xf16, 1>, %final: memref<2x4x3xf16, 1>) {
  // expected-error @+1 {{weight channels must match the input}}
  hip.varlen_causal_conv_with_state(%ctx)
      ins(%input, %weight, %cumulative, %state :
          memref<6x4xf16, 1>, memref<5x1x4xf16, 1>, memref<3xi32, 1>,
          memref<2x4x3xf16, 1>)
      outs(%output, %final : memref<6x4xf16, 1>, memref<2x4x3xf16, 1>)
      {activation = "silu"}
  return
}

// -----

func.func @varlen_state_len_mismatch(%ctx: !hip.context,
    %input: memref<6x4xf16, 1>, %weight: memref<4x1x4xf16, 1>,
    %cumulative: memref<3xi32, 1>, %state: memref<2x4x2xf16, 1>,
    %output: memref<6x4xf16, 1>, %final: memref<2x4x2xf16, 1>) {
  // expected-error @+1 {{initial_state dim 2 must be (kernel_size - 1) * dilation = 3}}
  hip.varlen_causal_conv_with_state(%ctx)
      ins(%input, %weight, %cumulative, %state :
          memref<6x4xf16, 1>, memref<4x1x4xf16, 1>, memref<3xi32, 1>,
          memref<2x4x2xf16, 1>)
      outs(%output, %final : memref<6x4xf16, 1>, memref<2x4x2xf16, 1>)
      {activation = "silu"}
  return
}
