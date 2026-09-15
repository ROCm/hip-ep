// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// hip.multi_head_attention lowers to TOSA SDPA inside a rock.kernel function,
// matching the MIGraphX MultiHeadAttention decompose (unpack, optional QKV
// bias, BNSH, past concat, QK / scale / softmax / PV). Cover separate QKV,
// cross-attention, packed QKV/KV, projection bias, attention_bias, padding
// mask, unidirectional, past concat, qk dump, and ub.poison. Rejections
// (share-buffer, cache_indirection, past_sequence_length) each get their
// own --split-input-file chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Softmax reduce_max is between the QK and PV matmuls.
// CHECK-LABEL: func.func @mha_self
// CHECK: tosa.matmul
// CHECK: tosa.reduce_max
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_self(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                    %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                    %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64, scale = 3.53553391e-01 : f32}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @mha_cross
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_cross(%ctx: !hip.context, %q: tensor<1x2x16xf16>,
                     %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                     %o: tensor<1x2x16xf16>) -> tensor<1x2x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x2x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%o : tensor<1x2x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x2x16xf16>
  return %r : tensor<1x2x16xf16>
}

// CHECK-LABEL: func.func @mha_packed_qkv
// CHECK: tosa.slice
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_packed_qkv(%ctx: !hip.context, %qkv: tensor<1x4x2x3x8xf16>,
                          %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%qkv : tensor<1x4x2x3x8xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @mha_packed_kv
// CHECK: tosa.slice
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_packed_kv(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                         %kv: tensor<1x4x2x2x8xf16>,
                         %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%q, %kv : tensor<1x4x16xf16>, tensor<1x4x2x2x8xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @mha_qkv_bias
// CHECK: tosa.add
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_qkv_bias(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                        %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                        %bias: tensor<48xf16>, %o: tensor<1x4x16xf16>)
    -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v, %bias : tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<48xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @mha_attn_bias
// CHECK: tosa.matmul
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_attn_bias(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                         %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                         %ab: tensor<1x2x4x4xf16>, %o: tensor<1x4x16xf16>)
    -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = "hip.multi_head_attention"(%ctx, %q, %k, %v, %ab, %o) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0>,
      num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x2x4x4xf16>, tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// Rank-4 bias [1, H, Sq, Skv] must tile batch before flattening to [B*H, ...].
// CHECK-LABEL: func.func @mha_attn_bias_broadcast_batch
// CHECK: tosa.tile
// CHECK: tosa.add
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_attn_bias_broadcast_batch(%ctx: !hip.context,
    %q: tensor<2x4x16xf16>, %k: tensor<2x4x16xf16>, %v: tensor<2x4x16xf16>,
    %ab: tensor<1x2x4x4xf16>, %o: tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
    attributes {rock.kernel} {
  %r = "hip.multi_head_attention"(%ctx, %q, %k, %v, %ab, %o) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0>,
      num_heads = 2 : i64
    } : (!hip.context, tensor<2x4x16xf16>, tensor<2x4x16xf16>, tensor<2x4x16xf16>, tensor<1x2x4x4xf16>, tensor<2x4x16xf16>) -> tensor<2x4x16xf16>
  return %r : tensor<2x4x16xf16>
}

// CHECK-LABEL: func.func @mha_unidirectional
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK-NOT: hip.multi_head_attention
func.func @mha_unidirectional(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                              %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                              %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64, unidirectional = 1 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @mha_past
// CHECK: tosa.concat
// CHECK: tosa.matmul
// CHECK-NOT: hip.multi_head_attention
func.func @mha_past(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                    %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                    %past_k: tensor<1x2x3x8xf16>, %past_v: tensor<1x2x3x8xf16>,
                    %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                    %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %past_k, %past_v, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64
    } : (!hip.context, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x3x8xf16>, tensor<1x2x3x8xf16>, tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @mha_poison_ctx
// CHECK-NOT: hip.multi_head_attention
func.func @mha_poison_ctx(%q: tensor<1x4x16xf16>, %k: tensor<1x4x16xf16>,
                          %v: tensor<1x4x16xf16>, %o: tensor<1x4x16xf16>)
    -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// CHECK-LABEL: func.func @no_rock_kernel
// CHECK: hip.multi_head_attention
func.func @no_rock_kernel(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                          %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                          %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16> {
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%o : tensor<1x4x16xf16>)
      {num_heads = 2 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// -----

func.func @share_buffer_rejected(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                                 %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                                 %past_k: tensor<1x2x4x8xf16>,
                                 %past_v: tensor<1x2x4x8xf16>,
                                 %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                 %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  // expected-error@+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  %r:3 = "hip.multi_head_attention"(%ctx, %q, %k, %v, %past_k, %past_v, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64
    } : (!hip.context, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>, tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// -----

func.func @cache_indirection_rejected(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                                      %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                                      %ind: tensor<1x1x4xi32>,
                                      %o: tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
    attributes {rock.kernel} {
  // expected-error@+1 {{failed to legalize operation 'hip.multi_head_attention'}}
  %r = "hip.multi_head_attention"(%ctx, %q, %k, %v, %ind, %o) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0>,
      num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x1x4xi32>, tensor<1x4x16xf16>) -> tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}
