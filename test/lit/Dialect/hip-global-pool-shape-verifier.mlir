// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @wrong_spatial_extent(
    %ctx: !hip.context,
    %input: tensor<2x3x8x8xf32>,
    %output: tensor<2x3x8x8xf32>) {
  // expected-error @+1 {{'hip.global_pool' op dim 2 of result mismatch: expected 1 [2, 3, 1, 1] but outs has 8 [2, 3, 8, 8]}}
  %result = hip.global_pool(%ctx)
      ins(%input : tensor<2x3x8x8xf32>)
      outs(%output : tensor<2x3x8x8xf32>)
      {mode = 0 : i64}
      : tensor<2x3x8x8xf32>
  return
}
