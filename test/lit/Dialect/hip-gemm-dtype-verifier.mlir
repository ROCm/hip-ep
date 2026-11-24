// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

func.func @mixed_inputs(%ctx: !hip.context, %a: memref<2x4xf16>, %b: memref<4x8xbf16>, %y: memref<2x8xf32>) {
  // expected-error @below {{GEMM inputs must have the same dtype}}
  hip.matmul(%ctx) ins(%a, %b : memref<2x4xf16>, memref<4x8xbf16>) outs(%y : memref<2x8xf32>)
  return
}
// -----
func.func @unsupported_output(%ctx: !hip.context, %a: memref<2x4xf16>, %b: memref<4x8xf16>, %y: memref<2x8xf64>) {
  // expected-error @below {{GEMM output must match the input dtype or be f32 for f16/bf16 inputs}}
  hip.matmul(%ctx) ins(%a, %b : memref<2x4xf16>, memref<4x8xf16>) outs(%y : memref<2x8xf64>)
  return
}
// -----
func.func @gemm_mixed_inputs(%ctx: !hip.context, %a: memref<2x4xf16>, %b: memref<4x8xf32>, %y: memref<2x8xf32>) {
  // expected-error @below {{GEMM inputs must have the same dtype}}
  hip.gemm(%ctx) ins(%a, %b : memref<2x4xf16>, memref<4x8xf32>) outs(%y : memref<2x8xf32>)
  return
}
// -----
func.func @gemm_wrong_bias(%ctx: !hip.context, %a: memref<2x4xbf16>, %b: memref<4x8xbf16>, %c: memref<8xbf16>, %y: memref<2x8xf32>) {
  // expected-error @below {{GEMM C must have the output dtype}}
  hip.gemm(%ctx) ins(%a, %b, %c : memref<2x4xbf16>, memref<4x8xbf16>, memref<8xbf16>) outs(%y : memref<2x8xf32>)
  return
}
// -----
func.func @accepted_bf16_score(%ctx: !hip.context, %a: memref<2x4xbf16>, %b: memref<4x8xbf16>, %y: memref<2x8xf32>) {
  hip.matmul(%ctx) ins(%a, %b : memref<2x4xbf16>, memref<4x8xbf16>) outs(%y : memref<2x8xf32>)
  return
}
// -----
func.func @accepted_fp64(%ctx: !hip.context, %a: memref<2x4xf64>, %b: memref<4x8xf64>, %y: memref<2x8xf64>) {
  hip.gemm(%ctx) ins(%a, %b : memref<2x4xf64>, memref<4x8xf64>) outs(%y : memref<2x8xf64>)
  return
}
