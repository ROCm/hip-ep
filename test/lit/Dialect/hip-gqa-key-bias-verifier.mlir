// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

// Generic form: the pretty ins() list cannot place an attention_bias without
// the optional operands before it. operandSegmentSizes order: handle, query,
// key, value, past_key, past_value, seqlens_k, total_seq_len, cos_cache,
// sin_cache, position_ids, attention_bias, head_sink, k_scale, v_scale,
// key_bias, output, present_key, present_value, output_qk.

func.func @key_bias_rank1(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                          %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                          %seqlens: tensor<1xi32>, %total: tensor<i32>,
                          %kb: tensor<4xf16>, %o: tensor<1x4x16xf16>,
                          %pk: tensor<1x2x4x8xf16>, %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{key_bias must be a rank-2 f16 [batch, total_seq_len] tensor or memref}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %kb, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}

// -----

func.func @key_bias_f32(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                        %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                        %seqlens: tensor<1xi32>, %total: tensor<i32>,
                        %kb: tensor<1x4xf32>, %o: tensor<1x4x16xf16>,
                        %pk: tensor<1x2x4x8xf16>, %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{key_bias must be a rank-2 f16 [batch, total_seq_len] tensor or memref}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %kb, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4xf32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}

// -----

func.func @key_bias_with_attention_bias(%ctx: !hip.context,
    %q: tensor<1x4x16xf16>, %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
    %seqlens: tensor<1xi32>, %total: tensor<i32>, %bias: tensor<1x1x4x4xf16>,
    %kb: tensor<1x4xf16>, %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
    %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{key_bias and attention_bias are mutually exclusive}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %bias, %kb, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x1x4x4xf16>, tensor<1x4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}

// -----

func.func @key_bias_no_causal(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                              %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                              %seqlens: tensor<1xi32>, %total: tensor<i32>,
                              %kb: tensor<1x4xf16>, %o: tensor<1x4x16xf16>,
                              %pk: tensor<1x2x4x8xf16>, %pv: tensor<1x2x4x8xf16>) {
  // expected-error @+1 {{key_bias requires causal attention (no_causal must be false)}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %kb, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return
}
