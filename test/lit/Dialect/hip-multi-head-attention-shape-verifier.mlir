// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @hidden_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x64xf16>,
    %output: tensor<1x8x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op dim 2 of result mismatch: expected 64 [1, 8, 64] but outs has 128 [1, 8, 128]}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x64xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>
  return
}

func.func @kv_sequence_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x15x128xf16>,
    %output: tensor<1x8x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op multi_head_attention K/V sequence extents must agree}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x15x128xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>
  return
}

// Individual optional destinations are legal independently of backend support.
func.func @present_key_only(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x8x128xf16>,
    %presentKey: tensor<1x8x16x16xf16>) {
  %result:2 = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output, %presentKey :
          tensor<1x8x128xf16>, tensor<1x8x16x16xf16>)
      {num_heads = 8 : i64}
      : tensor<1x8x128xf16>, tensor<1x8x16x16xf16>
  return
}

func.func @output_mismatch(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x7x128xf16>) {
  // expected-error @+1 {{'hip.multi_head_attention' op dim 1 of result mismatch: expected 8 [1, 8, 128] but outs has 7 [1, 7, 128]}}
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output : tensor<1x7x128xf16>)
      {num_heads = 8 : i64}
      : tensor<1x7x128xf16>
  return
}

func.func @nondefault_mask_filter_value(
    %ctx: !hip.context,
    %query: tensor<1x8x128xf16>,
    %key: tensor<1x16x128xf16>,
    %value: tensor<1x16x128xf16>,
    %output: tensor<1x8x128xf16>) {
  %result = hip.multi_head_attention(%ctx)
      ins(%query, %key, %value :
          tensor<1x8x128xf16>, tensor<1x16x128xf16>,
          tensor<1x16x128xf16>)
      outs(%output : tensor<1x8x128xf16>)
      {num_heads = 8 : i64, mask_filter_value = -5.0 : f32}
      : tensor<1x8x128xf16>
  return
}

func.func @different_value_width(
    %ctx: !hip.context, %q: tensor<1x8x128xf32>,
    %k: tensor<1x16x128xf32>, %v: tensor<1x16x64xf32>,
    %out: tensor<1x8x64xf32>) {
  %r = hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : tensor<1x8x128xf32>, tensor<1x16x128xf32>, tensor<1x16x64xf32>)
      outs(%out : tensor<1x8x64xf32>) {num_heads = 8 : i64} : tensor<1x8x64xf32>
  return
}

func.func @packed_qkv_invalid_packing(
    %ctx: !hip.context, %qkv: tensor<1x8x8x2x16xf16>,
    %out: tensor<1x8x128xf16>) {
  // expected-error @+1 {{packed QKV packing extent must be 3}}
  %r = hip.multi_head_attention(%ctx)
      ins(%qkv : tensor<1x8x8x2x16xf16>)
      outs(%out : tensor<1x8x128xf16>) {num_heads = 8 : i64} : tensor<1x8x128xf16>
  return
}

func.func @packed_qkv_hidden_overflow(
    %ctx: !hip.context, %qkv: tensor<1x1x2x3x4611686018427387904xf16>,
    %out: tensor<1x1x?xf16>) {
  // expected-error @+1 {{multi_head_attention hidden extent is out of range}}
  %r = hip.multi_head_attention(%ctx)
      ins(%qkv : tensor<1x1x2x3x4611686018427387904xf16>)
      outs(%out : tensor<1x1x?xf16>) {num_heads = 2 : i64} : tensor<1x1x?xf16>
  return
}

func.func @dynamic_value_cannot_specialize_output(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x16x128xf16>, %v: tensor<1x16x?xf16>,
    %out: tensor<1x8x64xf16>) {
  // expected-error @+1 {{output dimension 2 must remain dynamic because the corresponding source extent is dynamic}}
  %r = hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : tensor<1x8x128xf16>, tensor<1x16x128xf16>, tensor<1x16x?xf16>)
      outs(%out : tensor<1x8x64xf16>) {num_heads = 8 : i64} : tensor<1x8x64xf16>
  return
}

func.func @qk_independent_type(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x16x128xf16>, %v: tensor<1x16x64xf16>,
    %out: tensor<1x8x64xf16>, %qk: tensor<1x8x8x16xf32>) {
  %r:2 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %out, %qk) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1>,
      num_heads = 8 : i64
    } : (!hip.context, tensor<1x8x128xf16>, tensor<1x16x128xf16>, tensor<1x16x64xf16>, tensor<1x8x64xf16>, tensor<1x8x8x16xf32>) -> (tensor<1x8x64xf16>, tensor<1x8x8x16xf32>)
  return
}

func.func @qk_logical_sequence_mismatch(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x16x128xf16>, %v: tensor<1x16x64xf16>,
    %out: tensor<1x8x64xf16>, %qk: tensor<1x8x8x32xf32>) {
  // expected-error @+1 {{qk logical sequence extent must be 16, got 32}}
  %r:2 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %out, %qk) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1>,
      num_heads = 8 : i64
    } : (!hip.context, tensor<1x8x128xf16>, tensor<1x16x128xf16>, tensor<1x16x64xf16>, tensor<1x8x64xf16>, tensor<1x8x8x32xf32>) -> (tensor<1x8x64xf16>, tensor<1x8x8x32xf32>)
  return
}

func.func @cache_head_size_mismatch(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x16x128xf16>, %v: tensor<1x16x64xf16>,
    %out: tensor<1x8x64xf16>, %pk: tensor<1x8x16x8xf16>) {
  // expected-error @+1 {{present_key dimension 3 must be 16, got 8}}
  %r:2 = hip.multi_head_attention(%ctx)
      ins(%q, %k, %v : tensor<1x8x128xf16>, tensor<1x16x128xf16>, tensor<1x16x64xf16>)
      outs(%out, %pk : tensor<1x8x64xf16>, tensor<1x8x16x8xf16>)
      {num_heads = 8 : i64} : tensor<1x8x64xf16>, tensor<1x8x16x8xf16>
  return
}

// Sharing uses a runtime logical length; QK need not fill the cache capacity.
func.func @qk_shared_cache_logical_length(
    %ctx: !hip.context, %q: tensor<1x8x128xf16>,
    %k: tensor<1x8x128xf16>, %v: tensor<1x8x64xf16>,
    %pastk: tensor<1x8x64x16xf16>, %pastv: tensor<1x8x64x8xf16>,
    %pastlen: tensor<i32>, %out: tensor<1x8x64xf16>,
    %pk: tensor<1x8x64x16xf16>, %pv: tensor<1x8x64x8xf16>,
    %qk: tensor<1x8x8x16xf32>) {
  %r:4 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %pastk, %pastv, %pastlen, %out, %pk, %pv, %qk) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1>,
      num_heads = 8 : i64
    } : (!hip.context, tensor<1x8x128xf16>, tensor<1x8x128xf16>, tensor<1x8x64xf16>, tensor<1x8x64x16xf16>, tensor<1x8x64x8xf16>, tensor<i32>, tensor<1x8x64xf16>, tensor<1x8x64x16xf16>, tensor<1x8x64x8xf16>, tensor<1x8x8x16xf32>) -> (tensor<1x8x64xf16>, tensor<1x8x64x16xf16>, tensor<1x8x64x8xf16>, tensor<1x8x8x16xf32>)
  return
}
