// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// hip.gqa lowers to TOSA SDPA inside a rock.kernel function. Cover prefill
// MHA/GQA, decode+past concat, no_causal, bias, window, packed QKV, scale=0,
// RoPE, head_sink, INT8 KV, output_qk, the ub.poison context form, and a
// no-op without rock.kernel. Rejections (share-buffer, INT4, decode RoPE
// without position_ids) each get their own --split-input-file chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Pretty `ins(%q,%k,%v,%seqlens,%total)` is ambiguous: optional past_key/past_value
// greedily eat seqlens/total. No-past cases use generic form + operandSegmentSizes.
//
// Softmax (reduce_max … reciprocal) sits between the QK and PV matmuls.
// CHECK-LABEL: func.func @mha_prefill
// CHECK: tosa.matmul
// CHECK: tosa.select
// CHECK: tosa.reduce_max
// CHECK: tosa.sub
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum
// CHECK: tosa.reciprocal
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @mha_prefill(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                       %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                       %seqlens: tensor<1xi32>, %total: tensor<i32>,
                       %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                       %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, scale = 3.53553391e-01 : f32
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @gqa_prefill
// CHECK: tosa.tile
// CHECK: tosa.matmul
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @gqa_prefill(%ctx: !hip.context, %q: tensor<1x4x32xf16>,
                      %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                      %seqlens: tensor<1xi32>, %total: tensor<i32>,
                      %o: tensor<1x4x32xf16>, %pk: tensor<1x2x4x8xf16>,
                      %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x32xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 4 : i64, kv_num_heads = 2 : i64, scale = 3.53553391e-01 : f32
    } : (!hip.context, tensor<1x4x32xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x32xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x32xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x32xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @mha_prefill_poison
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @mha_prefill_poison(%q: tensor<1x4x16xf16>, %k: tensor<1x4x16xf16>,
                             %v: tensor<1x4x16xf16>, %seqlens: tensor<1xi32>,
                             %total: tensor<i32>, %o: tensor<1x4x16xf16>,
                             %pk: tensor<1x2x4x8xf16>, %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, scale = 3.53553391e-01 : f32
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @decode_past
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK: tosa.matmul
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @decode_past(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                       %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                       %past_k: tensor<1x2x3x8xf16>, %past_v: tensor<1x2x3x8xf16>,
                       %seqlens: tensor<1xi32>, %total: tensor<i32>,
                       %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                       %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = hip.gqa(%ctx) ins(%q, %k, %v, %past_k, %past_v, %seqlens, %total : tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x3x8xf16>, tensor<1x2x3x8xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
      {num_heads = 2 : i64, kv_num_heads = 2 : i64, scale = 3.53553391e-01 : f32}
      : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @no_causal
// CHECK: tosa.matmul
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @no_causal(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                     %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                     %seqlens: tensor<1xi32>, %total: tensor<i32>,
                     %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                     %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @bias
// CHECK: tosa.add
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @bias(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                %seqlens: tensor<1xi32>, %total: tensor<i32>,
                %bias: tensor<1x1x4x4xf16>,
                %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %bias, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x1x4x4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @window
// CHECK: tosa.select
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @window(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                  %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                  %seqlens: tensor<1xi32>, %total: tensor<i32>,
                  %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                  %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, local_window_size = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @packed_qkv
// CHECK: tosa.slice
// CHECK: tosa.matmul
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @packed_qkv(%ctx: !hip.context, %qkv: tensor<1x4x48xf16>,
                      %seqlens: tensor<1xi32>, %total: tensor<i32>,
                      %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                      %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %qkv, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64
    } : (!hip.context, tensor<1x4x48xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @scale_zero
// CHECK: tosa.mul
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @scale_zero(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                      %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                      %seqlens: tensor<1xi32>, %total: tensor<i32>,
                      %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                      %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, scale = 0.000000e+00 : f32, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// ORT prefill sentinel seqlens_k=-1 must not mask every key (kIdx > -1).
// CHECK-LABEL: func.func @seqlens_prefill_sentinel
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK: tosa.reduce_max
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @seqlens_prefill_sentinel(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                                   %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                                   %total: tensor<i32>,
                                   %o: tensor<1x4x16xf16>,
                                   %pk: tensor<1x2x4x8xf16>,
                                   %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %seqlens = arith.constant dense<-1> : tensor<1xi32>
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @rope_prefill
// CHECK: tosa.negate
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @rope_prefill(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                        %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                        %seqlens: tensor<1xi32>, %total: tensor<i32>,
                        %cos: tensor<4x4xf16>, %sin: tensor<4x4xf16>,
                        %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                        %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %cos, %sin, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<4x4xf16>, tensor<4x4xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @rope_prefill_with_position_ids
// CHECK: tosa.gather
// CHECK: tosa.negate
// CHECK-NOT: hip.gqa
func.func @rope_prefill_with_position_ids(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                                          %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                                          %seqlens: tensor<1xi32>, %total: tensor<i32>,
                                          %cos: tensor<4x4xf16>, %sin: tensor<4x4xf16>,
                                          %pos: tensor<1x4xi64>,
                                          %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                          %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %cos, %sin, %pos, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<4x4xf16>, tensor<4x4xf16>, tensor<1x4xi64>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// Decode RoPE applies only to the new Q/K tokens. Past cache is already
// rotated, so position_ids of shape [B, seqQ] are enough to gather.
// CHECK-LABEL: func.func @decode_rope_with_position_ids
// CHECK: tosa.gather
// CHECK: tosa.concat
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @decode_rope_with_position_ids(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                                         %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                                         %past_k: tensor<1x2x3x8xf16>,
                                         %past_v: tensor<1x2x3x8xf16>,
                                         %seqlens: tensor<1xi32>, %total: tensor<i32>,
                                         %cos: tensor<8x4xf16>, %sin: tensor<8x4xf16>,
                                         %pos: tensor<1x1xi64>,
                                         %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                         %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %past_k, %past_v, %seqlens, %total, %cos, %sin, %pos, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64
    } : (!hip.context, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x3x8xf16>, tensor<1x2x3x8xf16>, tensor<1xi32>, tensor<i32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<1x1xi64>, tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @head_sink
// CHECK: tosa.exp
// CHECK: tosa.add
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @head_sink(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                     %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                     %seqlens: tensor<1xi32>, %total: tensor<i32>,
                     %sink: tensor<2xf16>,
                     %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                     %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %sink, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, no_causal = true
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<2xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// CHECK-LABEL: func.func @int8_kv
// CHECK: tosa.cast
// CHECK: tosa.matmul
// CHECK-NOT: hip.gqa
func.func @int8_kv(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                   %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                   %past_k: tensor<1x2x3x8xi8>, %past_v: tensor<1x2x3x8xi8>,
                   %seqlens: tensor<1xi32>, %total: tensor<i32>,
                   %k_scale: tensor<1x2x1x8xf32>, %v_scale: tensor<1x2x1x8xf32>,
                   %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xi8>,
                   %pv: tensor<1x2x4x8xi8>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>)
    attributes {rock.kernel} {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %past_k, %past_v, %seqlens, %total, %k_scale, %v_scale, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, k_quant_type = "PER_CHANNEL", v_quant_type = "PER_CHANNEL", kv_cache_bit_width = 8 : i64
    } : (!hip.context, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x3x8xi8>, tensor<1x2x3x8xi8>, tensor<1xi32>, tensor<i32>, tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1x1x16xf16>, tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>) -> (tensor<1x1x16xf16>, tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>)
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xi8>, tensor<1x2x4x8xi8>
}

// CHECK-LABEL: func.func @no_rock_kernel
// CHECK: hip.gqa
func.func @no_rock_kernel(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                          %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                          %seqlens: tensor<1xi32>, %total: tensor<i32>,
                          %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                          %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) {
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, scale = 3.53553391e-01 : f32
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// -----

func.func @share_buffer_rejected(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                                 %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                                 %past_k: tensor<1x2x4x8xf16>,
                                 %past_v: tensor<1x2x4x8xf16>,
                                 %seqlens: tensor<1xi32>, %total: tensor<i32>,
                                 %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                 %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  // expected-error@+1 {{failed to legalize operation 'hip.gqa'}}
  %r:3 = hip.gqa(%ctx) ins(%q, %k, %v, %past_k, %past_v, %seqlens, %total : tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>, tensor<1xi32>, tensor<i32>)
      outs(%o, %pk, %pv : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
      {num_heads = 2 : i64, kv_num_heads = 2 : i64}
      : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// -----

func.func @int4_rejected(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                         %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>,
                         %seqlens: tensor<1xi32>, %total: tensor<i32>,
                         %k_scale: tensor<8xf16>, %v_scale: tensor<8xf16>,
                         %o: tensor<1x4x16xf16>, %pk: tensor<1x2x4x8xf16>,
                         %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  // expected-error@+1 {{failed to legalize operation 'hip.gqa'}}
  %r:3 = "hip.gqa"(%ctx, %q, %k, %v, %seqlens, %total, %k_scale, %v_scale, %o, %pk, %pv) {
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0>,
      num_heads = 2 : i64, kv_num_heads = 2 : i64, k_quant_type = "PER_CHANNEL", v_quant_type = "PER_CHANNEL", kv_cache_bit_width = 4 : i64
    } : (!hip.context, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1x4x16xf16>, tensor<1xi32>, tensor<i32>, tensor<8xf16>, tensor<8xf16>, tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>) -> (tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
  return %r#0, %r#1, %r#2 : tensor<1x4x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}

// -----

func.func @decode_rope_rejected(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                                %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>,
                                %past_k: tensor<1x2x3x8xf16>,
                                %past_v: tensor<1x2x3x8xf16>,
                                %seqlens: tensor<1xi32>, %total: tensor<i32>,
                                %cos: tensor<8x4xf16>, %sin: tensor<8x4xf16>,
                                %o: tensor<1x1x16xf16>, %pk: tensor<1x2x4x8xf16>,
                                %pv: tensor<1x2x4x8xf16>)
    -> (tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
    attributes {rock.kernel} {
  // expected-error@+1 {{failed to legalize operation 'hip.gqa'}}
  %r:3 = hip.gqa(%ctx) ins(%q, %k, %v, %past_k, %past_v, %seqlens, %total, %cos, %sin : tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x1x16xf16>, tensor<1x2x3x8xf16>, tensor<1x2x3x8xf16>, tensor<1xi32>, tensor<i32>, tensor<8x4xf16>, tensor<8x4xf16>)
      outs(%o, %pk, %pv : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>)
      {num_heads = 2 : i64, kv_num_heads = 2 : i64, do_rotary = 1 : i64}
      : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x16xf16>, tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16>
}
