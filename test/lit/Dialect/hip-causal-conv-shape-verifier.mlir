// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @wrong_output(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf16>, %output: memref<2x8x15xf16>,
    %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op dim 2 of result mismatch: expected 16}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<8x1x4xf16>)
      outs(%output, %present : memref<2x8x15xf16>, memref<2x8x3xf16>)
  return
}

func.func @wrong_present(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf16>, %output: memref<2x8x16xf16>,
    %present: memref<2x8x4xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op dim 2 of result mismatch: expected 3}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<8x1x4xf16>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x4xf16>)
  return
}

func.func @bad_weight_layout(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<7x2x4xf16>, %output: memref<2x8x16xf16>,
    %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op causal_conv_with_state weight channels must match input channels}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<7x2x4xf16>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x3xf16>)
  return
}

func.func @bad_past_state(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf16>, %bias: memref<8xf16>,
    %past: memref<1x8x3xf16>,
    %output: memref<2x8x16xf16>, %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op causal_conv_with_state past_state dimension 0 must match [B, C, K-1]}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight, %bias, %past :
          memref<2x8x16xf16>, memref<8x1x4xf16>, memref<8xf16>,
          memref<1x8x3xf16>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x3xf16>)
  return
}

func.func @unsupported_ndim(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf16>, %output: memref<2x8x16xf16>,
    %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op causal_conv_with_state runtime supports only ndim=1}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<8x1x4xf16>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x3xf16>)
      {ndim = 2 : i64}
  return
}

func.func @bad_activation(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf16>, %output: memref<2x8x16xf16>,
    %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op activation must be one of: none, silu, swish}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<8x1x4xf16>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x3xf16>)
      {activation = "relu"}
  return
}

func.func @mismatched_dtype(
    %ctx: !hip.context, %input: memref<2x8x16xf16>,
    %weight: memref<8x1x4xf32>, %output: memref<2x8x16xf16>,
    %present: memref<2x8x3xf16>) {
  // expected-error @+1 {{'hip.causal_conv_with_state' op all input and output element types must match}}
  hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : memref<2x8x16xf16>, memref<8x1x4xf32>)
      outs(%output, %present : memref<2x8x16xf16>, memref<2x8x3xf16>)
  return
}
