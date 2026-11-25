// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @mrope_bad_section_sum(%ctx: !hip.context,
    %x: tensor<1x1x512xf16>, %pos: tensor<3x1x1xi64>,
    %cos: tensor<64x32xf16>, %sin: tensor<64x32xf16>,
    %out: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  // expected-error @+1 {{mrope_section must sum to rotary_embedding_dim / 2 = 32, got 33}}
  %r = hip.mrotary_embedding(%ctx)
      ins(%x, %pos, %cos, %sin : tensor<1x1x512xf16>, tensor<3x1x1xi64>,
          tensor<64x32xf16>, tensor<64x32xf16>)
      outs(%out : tensor<1x1x512xf16>)
      {mrope_section = [11, 11, 11], num_heads = 2 : i64,
       rotary_embedding_dim = 64 : i64} : tensor<1x1x512xf16>
  return %r : tensor<1x1x512xf16>
}

// -----

func.func @mrope_rotary_too_wide(%ctx: !hip.context,
    %x: tensor<1x1x512xf16>, %pos: tensor<3x1x1xi64>,
    %cos: tensor<64x160xf16>, %sin: tensor<64x160xf16>,
    %out: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  // expected-error @+1 {{rotary_embedding_dim must be positive, even and at most the head size 256, got 320}}
  %r = hip.mrotary_embedding(%ctx)
      ins(%x, %pos, %cos, %sin : tensor<1x1x512xf16>, tensor<3x1x1xi64>,
          tensor<64x160xf16>, tensor<64x160xf16>)
      outs(%out : tensor<1x1x512xf16>)
      {mrope_section = [60, 50, 50], num_heads = 2 : i64,
       rotary_embedding_dim = 320 : i64} : tensor<1x1x512xf16>
  return %r : tensor<1x1x512xf16>
}

// -----

func.func @mrope_bad_layout(%ctx: !hip.context,
    %x: tensor<1x1x512xf16>, %pos: tensor<3x1x1xi64>,
    %cos: tensor<64x32xf16>, %sin: tensor<64x32xf16>,
    %out: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  // expected-error @+1 {{mrope_layout must be 0 or 1, got 2}}
  %r = hip.mrotary_embedding(%ctx)
      ins(%x, %pos, %cos, %sin : tensor<1x1x512xf16>, tensor<3x1x1xi64>,
          tensor<64x32xf16>, tensor<64x32xf16>)
      outs(%out : tensor<1x1x512xf16>)
      {mrope_layout = 2 : i64, mrope_section = [11, 11, 10],
       num_heads = 2 : i64, rotary_embedding_dim = 64 : i64}
      : tensor<1x1x512xf16>
  return %r : tensor<1x1x512xf16>
}

// -----

func.func @mrope_bad_position_ids(%ctx: !hip.context,
    %x: tensor<1x1x512xf16>, %pos: tensor<1x1xi64>,
    %cos: tensor<64x32xf16>, %sin: tensor<64x32xf16>,
    %out: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  // expected-error @+1 {{position_ids must be i64 [3, B, S], got 'tensor<1x1xi64>'}}
  %r = hip.mrotary_embedding(%ctx)
      ins(%x, %pos, %cos, %sin : tensor<1x1x512xf16>, tensor<1x1xi64>,
          tensor<64x32xf16>, tensor<64x32xf16>)
      outs(%out : tensor<1x1x512xf16>)
      {mrope_section = [11, 11, 10], num_heads = 2 : i64,
       rotary_embedding_dim = 64 : i64} : tensor<1x1x512xf16>
  return %r : tensor<1x1x512xf16>
}

// -----

func.func @mrope_bad_cache_width(%ctx: !hip.context,
    %x: tensor<1x1x512xf16>, %pos: tensor<3x1x1xi64>,
    %cos: tensor<64x128xf16>, %sin: tensor<64x128xf16>,
    %out: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
  // expected-error @+1 {{cos_cache and sin_cache must be 'f16' [max_position, 32], got 'tensor<64x128xf16>'}}
  %r = hip.mrotary_embedding(%ctx)
      ins(%x, %pos, %cos, %sin : tensor<1x1x512xf16>, tensor<3x1x1xi64>,
          tensor<64x128xf16>, tensor<64x128xf16>)
      outs(%out : tensor<1x1x512xf16>)
      {mrope_section = [11, 11, 10], num_heads = 2 : i64,
       rotary_embedding_dim = 64 : i64} : tensor<1x1x512xf16>
  return %r : tensor<1x1x512xf16>
}
