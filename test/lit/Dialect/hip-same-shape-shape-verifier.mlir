// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s


// data/output family with a second shaped input: bias cannot define the result.
func.func @bias_gelu_wrong_output(
    %ctx: !hip.context, %data: memref<2x8xf32, 1>,
    %bias: memref<8xf32, 1>, %output: memref<3x8xf32, 1>) {
  // expected-error @+1 {{dim 0 of result mismatch: expected 2}}
  hip.bias_gelu(%ctx)
    ins(%data, %bias : memref<2x8xf32, 1>, memref<8xf32, 1>)
    outs(%output : memref<3x8xf32, 1>)
  return
}

// -----

// x/y family with an unrelated scalar tensor operand.
func.func @cumsum_wrong_output(
    %ctx: !hip.context, %x: memref<2x3xf32, 1>,
    %axis: memref<i64, 1>, %y: memref<2x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.cumsum(%ctx)
    ins(%x, %axis : memref<2x3xf32, 1>, memref<i64, 1>)
    outs(%y : memref<2x4xf32, 1>)
  return
}

// -----

func.func @scatter_elements_wrong_output(
    %ctx: !hip.context, %data: memref<2x3xf32, 1>,
    %indices: memref<2x2xi64, 1>, %updates: memref<2x2xf32, 1>,
    %output: memref<2x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.scatter_elements(%ctx)
    ins(%data, %indices, %updates :
        memref<2x3xf32, 1>, memref<2x2xi64, 1>, memref<2x2xf32, 1>)
    outs(%output : memref<2x4xf32, 1>)
  return
}

// -----

func.func @scatter_nd_wrong_output(
    %ctx: !hip.context, %data: memref<4x5x6xf32, 1>,
    %indices: memref<2x1xi64, 1>, %updates: memref<2x5x6xf32, 1>,
    %output: memref<4x5x7xf32, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 6}}
  hip.scatter_nd(%ctx)
    ins(%data, %indices, %updates :
        memref<4x5x6xf32, 1>, memref<2x1xi64, 1>,
        memref<2x5x6xf32, 1>)
    outs(%output : memref<4x5x7xf32, 1>)
  return
}

// -----

// GatherElements follows indices, not data.
func.func @gather_elements_wrong_output(
    %ctx: !hip.context, %data: memref<8x3xf32, 1>,
    %indices: memref<2x3xi64, 1>, %output: memref<2x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.gather_elements(%ctx)
    ins(%data, %indices : memref<8x3xf32, 1>, memref<2x3xi64, 1>)
    outs(%output : memref<2x4xf32, 1>) {axis = 1 : i64}
  return
}

// -----

// RmsNorm retains its custom verifier and composes the named-input shape check
// after the structural contract.
func.func @rms_norm_wrong_output(
    %ctx: !hip.context, %input: memref<2x3xf16, 1>,
    %scale: memref<3xf16, 1>, %output: memref<2x4xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.rms_norm(%ctx)
    ins(%input, %scale : memref<2x3xf16, 1>, memref<3xf16, 1>)
    outs(%output : memref<2x4xf16, 1>)
    {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
  return
}

// -----

func.func @rope_wrong_output(
    %ctx: !hip.context, %input: memref<2x3x64xf16, 1>,
    %position_ids: memref<2x3xi64, 1>, %cos: memref<128x32xf16, 1>,
    %sin: memref<128x32xf16, 1>, %output: memref<2x4x64xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  hip.rope(%ctx)
    ins(%input, %position_ids, %cos, %sin :
        memref<2x3x64xf16, 1>, memref<2x3xi64, 1>,
        memref<128x32xf16, 1>, memref<128x32xf16, 1>)
    outs(%output : memref<2x4x64xf16, 1>)
    {interleaved = 0 : i64, num_heads = 1 : i64,
     rotary_embedding_dim = 64 : i64}
  return
}

// -----

func.func @qmoe_wrong_output(
    %ctx: !hip.context, %input: memref<1x2x16xf16, 1>,
    %router: memref<2x4xf16, 1>,
    %fc1w: memref<4x32x8xui8, 1>, %fc1s: memref<4x32x1xf16, 1>,
    %fc2w: memref<4x16x8xui8, 1>, %fc2s: memref<4x16x1xf16, 1>,
    %output: memref<1x3x16xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 2}}
  hip.qmoe(%ctx)
    ins(%input, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        memref<1x2x16xf16, 1>, memref<2x4xf16, 1>,
        memref<4x32x8xui8, 1>, memref<4x32x1xf16, 1>,
        memref<4x16x8xui8, 1>, memref<4x16x1xf16, 1>)
    outs(%output : memref<1x3x16xf16, 1>)
    {expert_weight_bits = 4 : i64, k = 2 : i64, block_size = 16 : i64,
     normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
     use_sparse_mixer = 0 : i64, activation_alpha = 1.0 : f32,
     activation_beta = 0.0 : f32, swiglu_limit = 7.0 : f32,
     activation_type = "swiglu"}
  return
}

// -----

// A generated verifier must include every shaped DPS input in the structural
// contract. Here the non-source data operand is a memref while indices/output
// are tensors.
func.func @gather_elements_mixed_non_source(
    %ctx: !hip.context, %data: memref<2x3xf32, 1>,
    %indices: tensor<2x3xi64>, %output: tensor<2x3xf32>) {
  // expected-error @+1 {{all data operands must be the same kind (all tensor or all memref)}}
  %result = hip.gather_elements(%ctx)
    ins(%data, %indices : memref<2x3xf32, 1>, tensor<2x3xi64>)
    outs(%output : tensor<2x3xf32>) {axis = 1 : i64} : tensor<2x3xf32>
  return
}

// -----

// The existing RmsNorm verifier must retain the same structural behavior before
// delegating to the shared same-shape check.
func.func @rms_norm_mixed_non_source(
    %ctx: !hip.context, %input: tensor<2x3xf16>,
    %scale: memref<3xf16, 1>, %output: tensor<2x3xf16>) {
  // expected-error @+1 {{all data operands must be the same kind (all tensor or all memref)}}
  %result = hip.rms_norm(%ctx)
    ins(%input, %scale : tensor<2x3xf16>, memref<3xf16, 1>)
    outs(%output : tensor<2x3xf16>)
    {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}
    : tensor<2x3xf16>
  return
}
