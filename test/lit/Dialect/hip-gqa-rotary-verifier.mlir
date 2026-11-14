// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

// cos/sin are [max_pos, rotary_dim / 2]. rotary_dim may be below head_dim
// (partial rotary) but not above it. Here head_dim = 16 / num_heads = 8.
// operandSegmentSizes order: handle, query, key, value, past_key, past_value,
// seqlens_k, total_seq_len, cos_cache, sin_cache, position_ids,
// attention_bias, head_sink, k_scale, v_scale, key_bias, output, present_key,
// present_value, output_qk.

func.func @partial_rotary_ok(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                             %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                             %seqlens: tensor<1xi32>, %total: tensor<i32>,
                             %cos: tensor<32x2xf16>, %sin: tensor<32x2xf16>,
                             %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                             %pv: tensor<1x2x4x8xf16>) {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %cos, %sin, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<32x2xf16>, tensor<32x2xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}

// -----

func.func @rotary_wider_than_head(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                                  %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                                  %seqlens: tensor<1xi32>, %total: tensor<i32>,
                                  %cos: tensor<32x8xf16>, %sin: tensor<32x8xf16>,
                                  %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                  %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{cos_cache width 8 gives rotary_dim 16, which must be in (0, head_dim = 8]}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %cos, %sin, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<32x8xf16>, tensor<32x8xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}

// -----

func.func @cos_sin_mismatch(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                            %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                            %seqlens: tensor<1xi32>, %total: tensor<i32>,
                            %cos: tensor<32x2xf16>, %sin: tensor<32x4xf16>,
                            %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                            %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{cos_cache and sin_cache shapes differ}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %cos, %sin, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<32x2xf16>, tensor<32x4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}
