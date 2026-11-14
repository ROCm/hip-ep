// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @head_ratio(%ctx: !hip.context, %q: memref<1x4x384xf16, 1>,
    %k: memref<1x2x16x64xf16, 1>, %v: memref<1x2x16x64xf16, 1>,
    %out: memref<1x4x384xf16, 1>) {
  // expected-error @+1 {{num_heads must be a positive multiple of kv_num_heads}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x384xf16, 1>, memref<1x2x16x64xf16, 1>,
          memref<1x2x16x64xf16, 1>)
      outs(%out : memref<1x4x384xf16, 1>)
      {num_heads = 6 : i64, kv_num_heads = 4 : i64, scale = 0.125 : f32}
  return
}

// -----

func.func @key_heads(%ctx: !hip.context, %q: memref<1x4x512xf16, 1>,
    %k: memref<1x4x16x64xf16, 1>, %v: memref<1x4x16x64xf16, 1>,
    %out: memref<1x4x512xf16, 1>) {
  // expected-error @+1 {{key and value must have kv_num_heads heads}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x512xf16, 1>, memref<1x4x16x64xf16, 1>,
          memref<1x4x16x64xf16, 1>)
      outs(%out : memref<1x4x512xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32}
  return
}

// -----

func.func @query_width(%ctx: !hip.context, %q: memref<1x4x256xf16, 1>,
    %k: memref<1x2x16x64xf16, 1>, %v: memref<1x2x16x64xf16, 1>,
    %out: memref<1x4x512xf16, 1>) {
  // expected-error @+1 {{query must be [B, S, num_heads * dk]}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x256xf16, 1>, memref<1x2x16x64xf16, 1>,
          memref<1x2x16x64xf16, 1>)
      outs(%out : memref<1x4x512xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32}
  return
}

// -----

func.func @output_width(%ctx: !hip.context, %q: memref<1x4x512xf16, 1>,
    %k: memref<1x2x16x64xf16, 1>, %v: memref<1x2x16x32xf16, 1>,
    %out: memref<1x4x512xf16, 1>) {
  // expected-error @+1 {{output must be [B, S, num_heads * dv]}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x512xf16, 1>, memref<1x2x16x64xf16, 1>,
          memref<1x2x16x32xf16, 1>)
      outs(%out : memref<1x4x512xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32}
  return
}

// -----

func.func @row_count(%ctx: !hip.context, %q: memref<1x4x512xf16, 1>,
    %k: memref<1x2x16x64xf16, 1>, %v: memref<1x2x32x64xf16, 1>,
    %out: memref<1x4x512xf16, 1>) {
  // expected-error @+1 {{key and value must have the same number of rows}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x512xf16, 1>, memref<1x2x16x64xf16, 1>,
          memref<1x2x32x64xf16, 1>)
      outs(%out : memref<1x4x512xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32}
  return
}

// -----

func.func @kv_seqlen_type(%ctx: !hip.context, %q: memref<1x4x512xf16, 1>,
    %k: memref<1x2x16x64xf16, 1>, %v: memref<1x2x16x64xf16, 1>,
    %lengths: memref<1xi32, 1>, %out: memref<1x4x512xf16, 1>) {
  // expected-error @+1 {{kv_seqlen must be [B] int64}}
  hip.kv_attention(%ctx)
      ins(%q, %k, %v : memref<1x4x512xf16, 1>, memref<1x2x16x64xf16, 1>,
          memref<1x2x16x64xf16, 1>)
      kv_seqlen(%lengths : memref<1xi32, 1>)
      outs(%out : memref<1x4x512xf16, 1>)
      {num_heads = 8 : i64, kv_num_heads = 2 : i64, scale = 0.125 : f32}
  return
}
