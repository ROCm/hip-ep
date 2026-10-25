// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @invalid_c(%ctx: !hip.context, %a: tensor<2x4xf32>,
                     %b: tensor<4x8xf32>, %c: tensor<3x8xf32>,
                     %out: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{gemm C dimension 3 is not broadcastable to output dimension 2}}
  %r = hip.gemm(%ctx)
    ins(%a, %b, %c : tensor<2x4xf32>, tensor<4x8xf32>, tensor<3x8xf32>)
    outs(%out : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

func.func @invalid_transpose(%ctx: !hip.context, %a: tensor<4x2xf32>,
                             %b: tensor<4x8xf32>, %out: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{gemm transA and transB must be 0 or 1}}
  %r = hip.gemm(%ctx)
    ins(%a, %b : tensor<4x2xf32>, tensor<4x8xf32>)
    outs(%out : tensor<2x8xf32>) {transA = 2 : i64} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

func.func @invalid_rank(%ctx: !hip.context, %a: tensor<1x2x4xf32>,
                        %b: tensor<4x8xf32>, %out: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error @+1 {{gemm A and B must be rank-2 tensors}}
  %r = hip.gemm(%ctx)
    ins(%a, %b : tensor<1x2x4xf32>, tensor<4x8xf32>)
    outs(%out : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

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
