// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: %if ring-gqa %{ hip-mlir-opt --split-input-file --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s --check-prefixes=CHECK,RING %} %else %{ hip-mlir-opt --split-input-file --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s --check-prefixes=CHECK,DEFAULT %}
// Ring is opt-in per proven-FP16 operation. All other calls retain the legacy
// ABI; this does not assert that upstream implements BF16 or softcap correctly.

module {
  func.func @ring_f16(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @ring_f16(
// RING: llvm.call @wrap_group_query_attention_ring_f16(
// DEFAULT: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @ring_dynamic_f16(%ctx: !hip.context,
                                %query: memref<?x?x6144xf16, 1>,
                                %key: memref<?x?x1024xf16, 1>,
                                %value: memref<?x?x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<?x?x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<?x?x6144xf16, 1>, memref<?x?x1024xf16, 1>, memref<?x?x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<?x?x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @ring_dynamic_f16(
// RING: llvm.call @wrap_group_query_attention_ring_f16(
// DEFAULT: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @bf16_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xbf16, 1>,
                                %key: memref<1x1x1024xbf16, 1>,
                                %value: memref<1x1x1024xbf16, 1>,
                                %past_key: memref<1x4x127x256xbf16, 1>,
                                %past_value: memref<1x4x127x256xbf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xbf16, 1>,
                                %present_key: memref<1x4x128x256xbf16, 1>,
                                %present_value: memref<1x4x128x256xbf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xbf16, 1>, memref<1x1x1024xbf16, 1>, memref<1x1x1024xbf16, 1>,
            memref<1x4x127x256xbf16, 1>, memref<1x4x127x256xbf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xbf16, 1>, memref<1x4x128x256xbf16, 1>, memref<1x4x128x256xbf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @bf16_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @fp32_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf32, 1>,
                                %key: memref<1x1x1024xf32, 1>,
                                %value: memref<1x1x1024xf32, 1>,
                                %past_key: memref<1x4x127x256xf32, 1>,
                                %past_value: memref<1x4x127x256xf32, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf32, 1>,
                                %present_key: memref<1x4x128x256xf32, 1>,
                                %present_value: memref<1x4x128x256xf32, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf32, 1>, memref<1x1x1024xf32, 1>, memref<1x1x1024xf32, 1>,
            memref<1x4x127x256xf32, 1>, memref<1x4x127x256xf32, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf32, 1>, memref<1x4x128x256xf32, 1>, memref<1x4x128x256xf32, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @fp32_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @mixed_key_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xbf16, 1>,
                                %value: memref<1x1x1024xbf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xbf16, 1>, memref<1x1x1024xbf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @mixed_key_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @mixed_cache_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xbf16, 1>,
                                %past_value: memref<1x4x127x256xbf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xbf16, 1>,
                                %present_value: memref<1x4x128x256xbf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xbf16, 1>, memref<1x4x127x256xbf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xbf16, 1>, memref<1x4x128x256xbf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @mixed_cache_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @mixed_output_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xbf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xbf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @mixed_output_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @positive_softcap_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 2.000000e+01 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @positive_softcap_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @negative_softcap_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = -2.000000e+01 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @negative_softcap_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @rotary_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 1 : i64, rotary_interleaved = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @rotary_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @window_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64, local_window_size = 128 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @window_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @bidirectional_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64, no_causal = true}
    return
  }
}

// CHECK-LABEL: llvm.func @bidirectional_fallback(
// CHECK: llvm.call @wrap_group_query_attention(

// -----

module {
  func.func @smooth_fallback(%ctx: !hip.context,
                                %query: memref<1x1x6144xf16, 1>,
                                %key: memref<1x1x1024xf16, 1>,
                                %value: memref<1x1x1024xf16, 1>,
                                %past_key: memref<1x4x127x256xf16, 1>,
                                %past_value: memref<1x4x127x256xf16, 1>,
                                %seqlens_k: memref<1x1xi32, 1>,
                                %total_seq_len: memref<i32, 1>,
                                %output: memref<1x1x6144xf16, 1>,
                                %present_key: memref<1x4x128x256xf16, 1>,
                                %present_value: memref<1x4x128x256xf16, 1>) {
    hip.gqa(%ctx)
        ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
            memref<1x1x6144xf16, 1>, memref<1x1x1024xf16, 1>, memref<1x1x1024xf16, 1>,
            memref<1x4x127x256xf16, 1>, memref<1x4x127x256xf16, 1>,
            memref<1x1xi32, 1>, memref<i32, 1>)
        outs(%output, %present_key, %present_value :
             memref<1x1x6144xf16, 1>, memref<1x4x128x256xf16, 1>, memref<1x4x128x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         scale = 0.0625 : f32, softcap = 0.000000e+00 : f32,
         do_rotary = 0 : i64, rotary_interleaved = 0 : i64, smooth_softmax = 1 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @smooth_fallback(
// CHECK: llvm.call @wrap_group_query_attention(
