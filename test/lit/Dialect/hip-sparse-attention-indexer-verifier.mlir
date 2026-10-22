// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @sai_too_many_blocks(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %sel: tensor<1x1x4099xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x4099xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{token_budget / compress_ratio must be at most 512, got 1024}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      outs(%sel, %present : tensor<1x1x4099xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 4096 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x4099xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x4099xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_bad_capacity(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %sel: tensor<1x1x2048xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2048xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{selected_indices dim 2 must be 2051, got 2048}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      outs(%sel, %present : tensor<1x1x2048xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2048xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2048xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_odd_rotary(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x63xf16>, %sin: tensor<8x63xf16>, %past: tensor<1x7x128xf16>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary width must be even and in (0, 128], got 63}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x63xf16>, tensor<8x63xf16>, tensor<1x7x128xf16>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_present_length(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %mask: tensor<1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x7x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x7x128xf16>) {
  // expected-error @+1 {{present_key dim 1 must be 8, got 7}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      mask(%mask : tensor<1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x7x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2051xi32>, tensor<1x7x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x7x128xf16>
}

// -----

func.func @sai_too_many_heads(%ctx: !hip.context,
    %qk: tensor<1x1x1280xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{query width must be (N + 1) * 128 with 1 <= N <= 8, got 1280}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x1280xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}
