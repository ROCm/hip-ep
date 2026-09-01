// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @gemm_static_match
// CHECK: hip.gemm
func.func @gemm_static_match(
    %ctx: !hip.context, %a: memref<2x4xf16, 1>,
    %b: memref<4x8xf16, 1>, %out: memref<2x8xf16, 1>) {
  hip.gemm(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>)
    outs(%out : memref<2x8xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @gemm_dynamic_k
// CHECK: hip.gemm
func.func @gemm_dynamic_k(
    %ctx: !hip.context, %a: memref<2x?xf16, 1>,
    %b: memref<?x8xf16, 1>, %out: memref<2x8xf16, 1>) {
  hip.gemm(%ctx)
    ins(%a, %b : memref<2x?xf16, 1>, memref<?x8xf16, 1>)
    outs(%out : memref<2x8xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @gemm_dynamic_k_one_sided
// CHECK: hip.gemm
func.func @gemm_dynamic_k_one_sided(
    %ctx: !hip.context, %a: memref<2x?xf16, 1>,
    %b: memref<4x8xf16, 1>, %out: memref<2x8xf16, 1>) {
  hip.gemm(%ctx)
    ins(%a, %b : memref<2x?xf16, 1>, memref<4x8xf16, 1>)
    outs(%out : memref<2x8xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @gemm_dynamic_k_transposed
// CHECK: hip.gemm
func.func @gemm_dynamic_k_transposed(
    %ctx: !hip.context, %a: memref<?x2xf16, 1>,
    %b: memref<8x?xf16, 1>, %out: memref<2x8xf16, 1>) {
  hip.gemm(%ctx)
    ins(%a, %b : memref<?x2xf16, 1>, memref<8x?xf16, 1>)
    outs(%out : memref<2x8xf16, 1>)
    {transA = 1 : i64, transB = 1 : i64}
  return
}

// -----

func.func @gemm_static_mismatch(
    %ctx: !hip.context, %a: memref<2x4xf16, 1>,
    %b: memref<5x8xf16, 1>, %out: memref<2x8xf16, 1>) {
  // expected-error @+1 {{gemm contraction dim mismatch: A has 4 but B has 5}}
  hip.gemm(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<5x8xf16, 1>)
    outs(%out : memref<2x8xf16, 1>)
  return
}
