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

func.func @sai_shared_present_length(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %psl: tensor<1xi32>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{present_key dim 1 must be 7, got 8}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      past_sequence_length(%psl : tensor<1xi32>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
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

// -----

func.func @sai_dynamic_capacity(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<8x64xf16>, %sin: tensor<8x64xf16>, %past: tensor<1x7x128xf16>,
    %sel: tensor<1x1x?xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x?xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{past_key, present_key and selected_indices must have a static last dim}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<8x64xf16>, tensor<8x64xf16>, tensor<1x7x128xf16>)
      outs(%sel, %present : tensor<1x1x?xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x?xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x?xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_ids_without_sections(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x4xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary_position_ids and rotary_sections must be given together}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_ids_batched_tables(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<1x16x4xf16>, %sin: tensor<1x16x4xf16>,
    %past: tensor<1x7x128xf16>, %ids: tensor<3x1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{cos_cache must be a rank 2 base table with rotary_position_ids}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<1x16x4xf16>, tensor<1x16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_sections_count(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x4xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary_sections must have 4 entries, got 3}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_section_out_of_range(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x4xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary_sections entries must be in [0, 3), got 3}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 3, 0>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_ids_type(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x4xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x1x8xi32>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary_position_ids must have element type 'i64', got 'i32'}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi32>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_ids_batch(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x4xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x2x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{rotary_position_ids dim 1 must be 1, got 2}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x4xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x2x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}

// -----

func.func @sai_ids_sin_width(%ctx: !hip.context,
    %qk: tensor<1x1x640xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %cos: tensor<16x4xf16>, %sin: tensor<16x8xf16>, %past: tensor<1x7x128xf16>,
    %ids: tensor<3x1x8xi64>,
    %sel: tensor<1x1x2051xi32>, %present: tensor<1x8x128xf16>)
    -> (tensor<1x1x2051xi32>, tensor<1x8x128xf16>) {
  // expected-error @+1 {{sin_cache dim 1 must be 4, got 8}}
  %r:2 = hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          tensor<1x1x640xf16>, tensor<128xf16>, tensor<128xf16>,
          tensor<16x4xf16>, tensor<16x8xf16>, tensor<1x7x128xf16>)
      rotary_position_ids(%ids : tensor<3x1x8xi64>)
      outs(%sel, %present : tensor<1x1x2051xi32>, tensor<1x8x128xf16>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
      : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
  return %r#0, %r#1 : tensor<1x1x2051xi32>, tensor<1x8x128xf16>
}
