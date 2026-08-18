// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --verify-diagnostics --split-input-file

func.func @valid_bf16(
    %ctx: !hip.context, %input: tensor<2x4xbf16>) {
  %output = tensor.empty() : tensor<2x4xbf16>
  %result = hip.miopen.softmax(%ctx)
      ins(%input : tensor<2x4xbf16>)
      outs(%output : tensor<2x4xbf16>) : tensor<2x4xbf16>
  return
}

// -----

func.func @mismatched_element_type(
    %ctx: !hip.context, %input: tensor<2x4xf16>) {
  %output = tensor.empty() : tensor<2x4xbf16>
  // expected-error @+1 {{input and output element types must match}}
  %result = hip.miopen.softmax(%ctx)
      ins(%input : tensor<2x4xf16>)
      outs(%output : tensor<2x4xbf16>) : tensor<2x4xbf16>
  return
}

// -----

func.func @unsupported_element_type(
    %ctx: !hip.context, %input: tensor<2x4xf64>) {
  %output = tensor.empty() : tensor<2x4xf64>
  // expected-error @+1 {{unsupported element type f64}}
  %result = hip.miopen.softmax(%ctx)
      ins(%input : tensor<2x4xf64>)
      outs(%output : tensor<2x4xf64>) : tensor<2x4xf64>
  return
}

// -----

func.func @rank_zero(%ctx: !hip.context, %input: tensor<f32>) {
  %output = tensor.empty() : tensor<f32>
  // expected-error @+1 {{requires positive-rank input and output}}
  %result = hip.miopen.softmax(%ctx)
      ins(%input : tensor<f32>)
      outs(%output : tensor<f32>) : tensor<f32>
  return
}

// -----

func.func @mismatched_shape(
    %ctx: !hip.context, %input: tensor<2x4xf32>) {
  %output = tensor.empty() : tensor<2x5xf32>
  // expected-error @+1 {{input and output dimensions must match at axis 1}}
  %result = hip.miopen.softmax(%ctx)
      ins(%input : tensor<2x4xf32>)
      outs(%output : tensor<2x5xf32>) : tensor<2x5xf32>
  return
}
