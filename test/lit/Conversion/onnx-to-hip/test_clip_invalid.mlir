// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a high-rank onnx.Clip whose bound is not a scalar is diagnosed rather
// than rewritten. PackBroadcastTo4D forwards Clip's min/max bounds unchanged
// while packing the data operand, so it only accepts a bound that is `none` or
// a rank-0 tensor. An unranked bound carries no rank to check, so the guard is
// written as a whitelist; otherwise it would be packed and the bound spliced
// into a rank-4 hip.max.
//
// This file uses `--verify-diagnostics` (not FileCheck) -- the conversion is
// expected to fail. Sister file `test_clip.mlir` covers the happy paths,
// including the rank-5 Clip that this pass exists to make legal.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file --verify-diagnostics %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // An unranked bound is unsupported by the elementwise ops either way; what
  // matters is that it is reported instead of silently packed.
  func.func @clip_5d_unranked_bound(%x: tensor<2x3x4x5x6xf32>, %lo: tensor<*xf32>)
      -> tensor<2x3x4x5x6xf32> {
    %n = "onnx.NoValue"() {value} : () -> none
    // Matching only the stable prefix: the trailing "but got ..." text is
    // formatted by MLIR and can change between versions.
    // expected-error @+1 {{'hip.max' op operand #2 must be ranked tensor or memref}}
    %y = "onnx.Clip"(%x, %lo, %n) : (tensor<2x3x4x5x6xf32>, tensor<*xf32>, none) -> tensor<2x3x4x5x6xf32>
    return %y : tensor<2x3x4x5x6xf32>
  }
}
