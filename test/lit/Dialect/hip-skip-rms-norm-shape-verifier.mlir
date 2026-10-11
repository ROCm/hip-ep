// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @wrong_skip_shape(
    %ctx: !hip.context, %input: memref<2x4x16xf16>,
    %skip: memref<2x5x16xf16>, %gamma: memref<16xf16>,
    %output: memref<2x4x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op skip_rms_norm skip dimension 1 must match input}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<2x4x16xf16>, memref<2x5x16xf16>, memref<16xf16>)
      outs(%output : memref<2x4x16xf16>) {epsilon = 1.0e-05 : f32}
  return
}

func.func @wrong_gamma(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf16>, %gamma: memref<15xf16>,
    %output: memref<8x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op skip_rms_norm gamma length must match the input's final dimension}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<8x16xf16>, memref<8x16xf16>, memref<15xf16>)
      outs(%output : memref<8x16xf16>) {epsilon = 1.0e-05 : f32}
  return
}

func.func @wrong_bias(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf16>, %gamma: memref<16xf16>,
    %bias: memref<8x16xf16>, %output: memref<8x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op skip_rms_norm bias must have rank 1}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma, %bias :
          memref<8x16xf16>, memref<8x16xf16>, memref<16xf16>,
          memref<8x16xf16>)
      outs(%output : memref<8x16xf16>) {epsilon = 1.0e-05 : f32}
  return
}

func.func @wrong_residual_shape(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf16>, %gamma: memref<16xf16>,
    %output: memref<8x16xf16>, %residual: memref<7x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op dim 0 of result mismatch: expected 8}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<8x16xf16>, memref<8x16xf16>, memref<16xf16>)
      outs(%output, %residual : memref<8x16xf16>, memref<7x16xf16>)
      {epsilon = 1.0e-05 : f32}
  return
}

func.func @too_many_outputs(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf16>, %gamma: memref<16xf16>,
    %out0: memref<8x16xf16>, %out1: memref<8x16xf16>,
    %out2: memref<8x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op expected 1 or 2 output buffers, got 3}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<8x16xf16>, memref<8x16xf16>, memref<16xf16>)
      outs(%out0, %out1, %out2 :
           memref<8x16xf16>, memref<8x16xf16>, memref<8x16xf16>)
      {epsilon = 1.0e-05 : f32}
  return
}

func.func @mismatched_dtype(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf32>, %gamma: memref<16xf16>,
    %output: memref<8x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op all input and output element types must match}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<8x16xf16>, memref<8x16xf32>, memref<16xf16>)
      outs(%output : memref<8x16xf16>) {epsilon = 1.0e-05 : f32}
  return
}

func.func @negative_epsilon(
    %ctx: !hip.context, %input: memref<8x16xf16>,
    %skip: memref<8x16xf16>, %gamma: memref<16xf16>,
    %output: memref<8x16xf16>) {
  // expected-error @+1 {{'hip.skip_rms_norm' op epsilon must be non-negative}}
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma :
          memref<8x16xf16>, memref<8x16xf16>, memref<16xf16>)
      outs(%output : memref<8x16xf16>) {epsilon = -1.0 : f32}
  return
}
