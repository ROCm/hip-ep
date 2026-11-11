// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @dsa_bad_head_size(%ctx: !hip.context,
    %q: tensor<1x1x256xf16>, %k: tensor<1x1x64xf16>, %v: tensor<1x1x64xf16>,
    %pk: tensor<1x1x32x64xf16>, %pv: tensor<1x1x32x64xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x256xf16>, %ok: tensor<1x1x32x64xf16>,
    %ov: tensor<1x1x32x64xf16>)
    -> (tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>) {
  // expected-error @+1 {{head size must be 128 or 256, got 64}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x256xf16>)
      kv(%k, %v : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      cache(%pk, %pv : tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>,
           tensor<1x1x32x64xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>
}

// -----

func.func @dsa_too_many_heads_per_group(%ctx: !hip.context,
    %q: tensor<1x1x4352xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x4352xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{num_heads / kv_num_heads must be at most 16, got 34}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x4352xf16>)
      kv(%k, %v : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 34 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_present_capacity_mismatch(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x33x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{present_key dim 2 must be 32, got 33}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x512xf16>)
      kv(%k, %v : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_bad_indices_type(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi64>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{selected_indices must have element type 'i32', got 'i64'}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x512xf16>)
      kv(%k, %v : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi64>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_pair_type(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %psl: tensor<1xi64>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{past_sequence_length must have element type 'i32', got 'i64'}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x512xf16>)
      kv(%k, %v : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      pair(%psl : tensor<1xi64>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_negative_scale(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{scale must be finite and non-negative}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%q : tensor<1x1x512xf16>)
      kv(%k, %v : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = -1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

// A packed query holds N + 2 * kvN heads: 4 + 2 heads of 128 is 768 wide.
func.func @dsa_packed_bad_width(%ctx: !hip.context,
    %qkv: tensor<1x1x700xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{query must be rank 3 with a static last dim that is a multiple of num_heads + 2 * kv_num_heads}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%qkv : tensor<1x1x700xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_gate_shape(%ctx: !hip.context,
    %qkv: tensor<1x1x768xf16>, %g: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{gate dim 2 must be 512, got 128}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%qkv : tensor<1x1x768xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      gate(%g : tensor<1x1x128xf16>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_prologue_attrs_without_inputs(%ctx: !hip.context,
    %qkv: tensor<1x1x768xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{q_norm_weight, k_norm_weight, position_ids, cos_cache, sin_cache, norm_epsilon, mrope_section and mrope_layout must be given together}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%qkv : tensor<1x1x768xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32,
       norm_epsilon = 9.99999997E-7 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

// Rotary width 2 * (24 + 24 + 24) = 144 exceeds the head size.
func.func @dsa_prologue_rotary_too_wide(%ctx: !hip.context,
    %qkv: tensor<1x1x768xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %pos: tensor<3x1x1xi64>, %cos: tensor<64x72xf16>, %sin: tensor<64x72xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{rotary width 2 * sum(mrope_section) must be in (0, head size], got 144}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%qkv : tensor<1x1x768xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      prologue(%qw, %kw, %pos, %cos, %sin : tensor<128xf16>, tensor<128xf16>,
               tensor<3x1x1xi64>, tensor<64x72xf16>, tensor<64x72xf16>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32,
       norm_epsilon = 9.99999997E-7 : f32, mrope_section = array<i64: 24, 24, 24>,
       mrope_layout = 1 : i64}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_prologue_cos_width(%ctx: !hip.context,
    %qkv: tensor<1x1x768xf16>, %qw: tensor<128xf16>, %kw: tensor<128xf16>,
    %pos: tensor<3x1x1xi64>, %cos: tensor<64x16xf16>, %sin: tensor<64x32xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{cos_cache dim 1 must be 32, got 16}}
  %r:3 = hip.dynamic_sparse_attention(%ctx) ins(%qkv : tensor<1x1x768xf16>)
      cache(%pk, %pv : tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>)
      select(%idx, %cnt, %seq : tensor<1x8xi32>, tensor<1xi32>, tensor<1xi32>)
      prologue(%qw, %kw, %pos, %cos, %sin : tensor<128xf16>, tensor<128xf16>,
               tensor<3x1x1xi64>, tensor<64x16xf16>, tensor<64x32xf16>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32,
       norm_epsilon = 9.99999997E-7 : f32, mrope_section = array<i64: 11, 11, 10>,
       mrope_layout = 1 : i64}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}
