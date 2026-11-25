// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

// Valid forms: separate key/value, packed QKV, and rotary tables.
func.func @paged_valid(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

func.func @paged_valid_packed(%ctx: !hip.context,
    %q: tensor<4x128xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x128xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

func.func @paged_valid_rotary(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %cos: tensor<32x8xf16>, %sin: tensor<32x8xf16>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      cos(%cos : tensor<32x8xf16>)
      sin(%sin : tensor<32x8xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64, rotary_interleaved = 1 : i64, local_window_size = 8 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_key_without_value(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{key and value must be both given or both absent}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_bad_head_ratio(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{num_heads (3) must be a positive multiple of kv_num_heads (2)}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 3 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_bad_query_width(%ctx: !hip.context,
    %q: tensor<4x62xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{query must be rank 2 with a static, positive dim 1 that is a multiple of num_heads}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x62xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_bad_cache_head_size(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x8xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{key_cache dim 3 must be 16, got 8}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x8xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_bad_cu_length(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<3xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{cumulative_sequence_length dim 0 must be 2, got 3}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<3xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_block_table_not_i32(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi64>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{block_table must have element type 'i32', got 'i64'}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi64>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_query_f32(%ctx: !hip.context,
    %q: tensor<4x64xf32>,
    %k: tensor<4x32xf32>, %v: tensor<4x32xf32>,
    %kc: tensor<8x16x2x16xf32>, %vc: tensor<8x16x2x16xf32>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf32>, %kco: tensor<8x16x2x16xf32>, %vco: tensor<8x16x2x16xf32>)
    -> (tensor<4x64xf32>, tensor<8x16x2x16xf32>, tensor<8x16x2x16xf32>) {
  // expected-error @+1 {{query must be f16, got 'f32'}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf32>, tensor<8x16x2x16xf32>,
          tensor<8x16x2x16xf32>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf32>)
      value(%v : tensor<4x32xf32>)
      outs(%out, %kco, %vco : tensor<4x64xf32>, tensor<8x16x2x16xf32>,
           tensor<8x16x2x16xf32>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf32>, tensor<8x16x2x16xf32>, tensor<8x16x2x16xf32>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf32>, tensor<8x16x2x16xf32>, tensor<8x16x2x16xf32>
}

// -----

func.func @paged_cache_out_shape(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<4x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<4x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{key_cache_out dim 0 must be 8, got 4}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<4x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xf16>, tensor<4x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<4x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_rotary_without_tables(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{do_rotary requires cos_cache and sin_cache}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_bad_cos_width(%ctx: !hip.context,
    %q: tensor<4x64xf16>,
    %k: tensor<4x32xf16>, %v: tensor<4x32xf16>,
    %kc: tensor<8x16x2x16xf16>, %vc: tensor<8x16x2x16xf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %cos: tensor<32x16xf16>, %sin: tensor<32x8xf16>,
    %out: tensor<4x64xf16>, %kco: tensor<8x16x2x16xf16>, %vco: tensor<8x16x2x16xf16>)
    -> (tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>) {
  // expected-error @+1 {{cos_cache dim 1 must be 8, got 16}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
          tensor<8x16x2x16xf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xf16>)
      value(%v : tensor<4x32xf16>)
      cos(%cos : tensor<32x16xf16>)
      sin(%sin : tensor<32x8xf16>)
      outs(%out, %kco, %vco : tensor<4x64xf16>, tensor<8x16x2x16xf16>,
           tensor<8x16x2x16xf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64}
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xf16>, tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>
}

// -----

func.func @paged_query_bf16(%ctx: !hip.context,
    %q: tensor<4x64xbf16>,
    %k: tensor<4x32xbf16>, %v: tensor<4x32xbf16>,
    %kc: tensor<8x16x2x16xbf16>, %vc: tensor<8x16x2x16xbf16>,
    %cu: tensor<2xi32>, %past: tensor<1xi32>, %bt: tensor<1x8xi32>,
    %out: tensor<4x64xbf16>, %kco: tensor<8x16x2x16xbf16>, %vco: tensor<8x16x2x16xbf16>)
    -> (tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>) {
  // expected-error @+1 {{query must be f16, got 'bf16'}}
  %r:3 = hip.paged_attention(%ctx)
      ins(%q, %kc, %vc, %cu, %past, %bt : tensor<4x64xbf16>, tensor<8x16x2x16xbf16>,
          tensor<8x16x2x16xbf16>, tensor<2xi32>, tensor<1xi32>, tensor<1x8xi32>)
      key(%k : tensor<4x32xbf16>)
      value(%v : tensor<4x32xbf16>)
      outs(%out, %kco, %vco : tensor<4x64xbf16>, tensor<8x16x2x16xbf16>,
           tensor<8x16x2x16xbf16>)
      {num_heads = 4 : i64, kv_num_heads = 2 : i64}
      : tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>
  return %r#0, %r#1, %r#2
      : tensor<4x64xbf16>, tensor<8x16x2x16xbf16>, tensor<8x16x2x16xbf16>
}
