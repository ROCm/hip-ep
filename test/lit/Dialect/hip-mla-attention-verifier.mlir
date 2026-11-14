// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @scales_without_block(%ctx: !hip.context,
    %q: memref<1x1x4096xf16, 1>, %cache: memref<1x1x8x320xf16, 1>,
    %kb: memref<256x2048xf16, 1>, %vb: memref<256x4096xf16, 1>,
    %ks: memref<2048xf16, 1>, %vs: memref<4096xf16, 1>,
    %out: memref<1x1x4096xf16, 1>) {
  // expected-error @+1 {{k_b_scales and v_b_scales are required exactly when block_size > 0}}
  hip.mla_attention(%ctx)
      ins(%q, %cache, %kb, %vb : memref<1x1x4096xf16, 1>,
          memref<1x1x8x320xf16, 1>, memref<256x2048xf16, 1>,
          memref<256x4096xf16, 1>)
      k_b_scales(%ks : memref<2048xf16, 1>)
      v_b_scales(%vs : memref<4096xf16, 1>)
      outs(%out : memref<1x1x4096xf16, 1>)
      {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
       qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32}
  return
}

// -----

func.func @cache_width(%ctx: !hip.context,
    %q: memref<1x1x4096xf16, 1>, %cache: memref<1x1x8x256xf16, 1>,
    %kb: memref<256x2048xf16, 1>, %vb: memref<256x4096xf16, 1>,
    %out: memref<1x1x4096xf16, 1>) {
  // expected-error @+1 {{cache must be [B, 1, T, kv_lora_rank + qk_rope_dim]}}
  hip.mla_attention(%ctx)
      ins(%q, %cache, %kb, %vb : memref<1x1x4096xf16, 1>,
          memref<1x1x8x256xf16, 1>, memref<256x2048xf16, 1>,
          memref<256x4096xf16, 1>)
      outs(%out : memref<1x1x4096xf16, 1>)
      {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
       qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32}
  return
}

// -----

func.func @int4_weight_shape(%ctx: !hip.context,
    %q: memref<1x1x4096xf16, 1>, %cache: memref<1x1x8x320xf16, 1>,
    %kb: memref<2048x4x32xui8, 1>, %vb: memref<4096x2x64xui8, 1>,
    %ks: memref<4096xf16, 1>, %vs: memref<8192xf16, 1>,
    %out: memref<1x1x4096xf16, 1>) {
  // expected-error @+1 {{k_b/v_b must be fp16 [kv_lora_rank, num_heads * dim] or int4-packed}}
  hip.mla_attention(%ctx)
      ins(%q, %cache, %kb, %vb : memref<1x1x4096xf16, 1>,
          memref<1x1x8x320xf16, 1>, memref<2048x4x32xui8, 1>,
          memref<4096x2x64xui8, 1>)
      k_b_scales(%ks : memref<4096xf16, 1>)
      v_b_scales(%vs : memref<8192xf16, 1>)
      outs(%out : memref<1x1x4096xf16, 1>)
      {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
       qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32,
       block_size = 128 : i64}
  return
}

// -----

func.func @kv_seqlen_type(%ctx: !hip.context,
    %q: memref<1x1x4096xf16, 1>, %cache: memref<1x1x8x320xf16, 1>,
    %kb: memref<256x2048xf16, 1>, %vb: memref<256x4096xf16, 1>,
    %lengths: memref<1xi32, 1>, %out: memref<1x1x4096xf16, 1>) {
  // expected-error @+1 {{kv_seqlen must be [B] int64}}
  hip.mla_attention(%ctx)
      ins(%q, %cache, %kb, %vb : memref<1x1x4096xf16, 1>,
          memref<1x1x8x320xf16, 1>, memref<256x2048xf16, 1>,
          memref<256x4096xf16, 1>)
      kv_seqlen(%lengths : memref<1xi32, 1>)
      outs(%out : memref<1x1x4096xf16, 1>)
      {num_heads = 32 : i64, kv_lora_rank = 256 : i64, qk_nope_dim = 64 : i64,
       qk_rope_dim = 64 : i64, v_dim = 128 : i64, scale = 0.195 : f32}
  return
}
