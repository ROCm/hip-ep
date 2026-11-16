// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @runtime_scales_without_launch(%ctx: !hip.context, %s: f64,
    %x: memref<1x1x3xf32, 1>, %y: memref<1x1x4xf32, 1>) {
  // expected-error @+1 {{runtime_scales requires prefix_count and spatial_rank}}
  hip.resize(%ctx) ins(%x : memref<1x1x3xf32, 1>) scales(%s : f64)
      outs(%y : memref<1x1x4xf32, 1>)
      {mode = 1, coord_transform = 0, nearest_mode = 0}
  return
}

// -----

func.func @runtime_scales_count_mismatch(%ctx: !hip.context, %s: f64,
    %x: memref<1x1x3x3xf32, 1>, %y: memref<1x1x4x4xf32, 1>) {
  // expected-error @+1 {{expected one runtime scale per window axis (2), got 1}}
  hip.resize(%ctx) ins(%x : memref<1x1x3x3xf32, 1>) scales(%s : f64)
      outs(%y : memref<1x1x4x4xf32, 1>)
      {mode = 1, coord_transform = 0, nearest_mode = 0,
       prefix_count = 2, spatial_rank = 2}
  return
}

// -----

func.func @runtime_scales_with_constant_scales(%ctx: !hip.context, %s: f64,
    %x: memref<1x1x3xf32, 1>, %y: memref<1x1x4xf32, 1>) {
  // expected-error @+1 {{runtime_scales and scales are mutually exclusive}}
  hip.resize(%ctx) ins(%x : memref<1x1x3xf32, 1>) scales(%s : f64)
      outs(%y : memref<1x1x4xf32, 1>)
      {mode = 1, coord_transform = 0, nearest_mode = 0,
       prefix_count = 2, spatial_rank = 1, scales = [1.0, 1.0, 1.5]}
  return
}
