// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid(%ctx: !hip.context,
                 %input: memref<2x3x4xf16, 1>,
                 %scale: memref<3x4xf16, 1>,
                 %output: memref<2x3x4xf16, 1>,
                 %mean: memref<2x1x1xf32, 1>,
                 %inv: memref<2x1x1xf32, 1>) {
  hip.layer_norm(%ctx)
    ins(%input, %scale : memref<2x3x4xf16, 1>, memref<3x4xf16, 1>)
    outs(%output, %mean, %inv : memref<2x3x4xf16, 1>,
                                      memref<2x1x1xf32, 1>,
                                      memref<2x1x1xf32, 1>)
    {axis = 1 : i64, stash_type = 1 : i64}
  return
}

// -----

func.func @wrong_stats_shape(
    %ctx: !hip.context,
    %input: memref<2x3x4xf16, 1>,
    %scale: memref<4xf16, 1>,
    %output: memref<2x3x4xf16, 1>,
    %mean: memref<2x3x4xf32, 1>) {
  // expected-error @+1 {{output #1 dimension 2 must be 1}}
  hip.layer_norm(%ctx)
    ins(%input, %scale : memref<2x3x4xf16, 1>, memref<4xf16, 1>)
    outs(%output, %mean : memref<2x3x4xf16, 1>, memref<2x3x4xf32, 1>)
    {axis = -1 : i64, stash_type = 1 : i64}
  return
}

// -----

func.func @wrong_scale_shape(
    %ctx: !hip.context,
    %input: memref<2x3x4xf16, 1>,
    %scale: memref<4xf16, 1>,
    %output: memref<2x3x4xf16, 1>) {
  // expected-error @+1 {{runtime requires scale rank to equal the normalized suffix rank}}
  hip.layer_norm(%ctx)
    ins(%input, %scale : memref<2x3x4xf16, 1>, memref<4xf16, 1>)
    outs(%output : memref<2x3x4xf16, 1>)
    {axis = 1 : i64, stash_type = 1 : i64}
  return
}

// -----

func.func @wrong_stats_type(
    %ctx: !hip.context,
    %input: memref<2x4xf16, 1>,
    %scale: memref<4xf16, 1>,
    %output: memref<2x4xf16, 1>,
    %mean: memref<2x1xf16, 1>) {
  // expected-error @+1 {{stats output element type must match stash_type}}
  hip.layer_norm(%ctx)
    ins(%input, %scale : memref<2x4xf16, 1>, memref<4xf16, 1>)
    outs(%output, %mean : memref<2x4xf16, 1>, memref<2x1xf16, 1>)
    {axis = -1 : i64, stash_type = 1 : i64}
  return
}

// -----

func.func @invalid_axis(%ctx: !hip.context,
                        %input: memref<2x4xf16, 1>,
                        %scale: memref<4xf16, 1>,
                        %output: memref<2x4xf16, 1>) {
  // expected-error @+1 {{axis must be in [-rank, rank)}}
  hip.layer_norm(%ctx)
    ins(%input, %scale : memref<2x4xf16, 1>, memref<4xf16, 1>)
    outs(%output : memref<2x4xf16, 1>)
    {axis = 2 : i64, stash_type = 1 : i64}
  return
}
