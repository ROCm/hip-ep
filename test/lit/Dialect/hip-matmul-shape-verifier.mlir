// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for the static shape verifier on `hip.matmul`.
//
// `hip.matmul` is a destination-passing-style op whose contract is:
//   A: [..., M, K]
//   B: [..., K, N]
//   output: [broadcast(A.batch, B.batch), M, N]
//
// The verifier (HipShapeUtils::verifyHipOpShape backed by
// HipShapeUtils::inferContractionShape) accepts kDynamic on either side as
// a wildcard and rejects any pair of statically-known dims that disagrees.
//
// The verifier ALSO inherits the cross-cutting all-tensor-or-all-memref
// check from verifyDpsComputeOp; that path is exercised by other LIT
// suites, so this file focuses on the contraction-specific checks.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// --- Positive: 2D matmul with all-static shapes accepts. ---
// CHECK-LABEL: func.func @matmul_2d_static
// CHECK:         hip.matmul
func.func @matmul_2d_static(%ctx: !hip.context,
                            %a: memref<128x4096xf32, 1>,
                            %b: memref<4096x1024xf32, 1>,
                            %c: memref<128x1024xf32, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<128x4096xf32, 1>, memref<4096x1024xf32, 1>)
    outs(%c : memref<128x1024xf32, 1>)
  return
}

// -----

// --- Positive: 3D batched matmul with broadcast batch ([2] x [] -> [2]). ---
// CHECK-LABEL: func.func @matmul_3d_broadcast
// CHECK:         hip.matmul
func.func @matmul_3d_broadcast(%ctx: !hip.context,
                               %a: memref<2x4x8xf16, 1>,
                               %b: memref<8x16xf16, 1>,
                               %c: memref<2x4x16xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
  return
}

// -----

// --- Positive: kDynamic on every batch dim — wildcard, no error. ---
// CHECK-LABEL: func.func @matmul_dynamic_batch
// CHECK:         hip.matmul
func.func @matmul_dynamic_batch(%ctx: !hip.context,
                                %a: memref<?x4x8xf16, 1>,
                                %b: memref<8x16xf16, 1>,
                                %c: memref<?x4x16xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<?x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<?x4x16xf16, 1>)
  return
}

// -----

// --- Positive: kDynamic on contraction K — wildcard, no error. ---
// CHECK-LABEL: func.func @matmul_dynamic_k
// CHECK:         hip.matmul
func.func @matmul_dynamic_k(%ctx: !hip.context,
                            %a: memref<2x?xf16, 1>,
                            %b: memref<?x8xf16, 1>,
                            %c: memref<2x8xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x?xf16, 1>, memref<?x8xf16, 1>)
    outs(%c : memref<2x8xf16, 1>)
  return
}

// -----

// --- Negative: contraction K mismatch is rejected. ---
func.func @matmul_k_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<8x16xf16, 1>,
                             %c: memref<2x16xf16, 1>) {
  // expected-error @below {{matmul contraction dim mismatch: A.shape[-1]=4 vs B.shape[-2]=8}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<2x16xf16, 1>)
  return
}

// -----

// --- Negative: M (output dim 0 in 2D case) mismatches outs. ---
func.func @matmul_m_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<4x8xf16, 1>,
                             %c: memref<3x8xf16, 1>) {
  // expected-error @below {{dim 0 of result #0 mismatch: expected 2}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>)
    outs(%c : memref<3x8xf16, 1>)
  return
}

// -----

// --- Negative: N (output dim 1 in 2D case) mismatches outs. ---
func.func @matmul_n_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<4x8xf16, 1>,
                             %c: memref<2x9xf16, 1>) {
  // expected-error @below {{dim 1 of result #0 mismatch: expected 8}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>)
    outs(%c : memref<2x9xf16, 1>)
  return
}

// -----

// --- Negative: batch dim broadcast failure ([2] vs [3], neither is 1). ---
func.func @matmul_batch_broadcast_fail(%ctx: !hip.context,
                                       %a: memref<2x4x8xf16, 1>,
                                       %b: memref<3x8x16xf16, 1>,
                                       %c: memref<2x4x16xf16, 1>) {
  // expected-error @below {{matmul batch dim broadcast failure}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<3x8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
  return
}

// -----

// --- Negative: rank of outs disagrees with the expected output rank
//     (broadcast(A.batch=[2], B.batch=[]) gives rank 3, but outs is rank 2). ---
func.func @matmul_rank_mismatch(%ctx: !hip.context,
                                %a: memref<2x4x8xf16, 1>,
                                %b: memref<8x16xf16, 1>,
                                %c: memref<4x16xf16, 1>) {
  // expected-error @below {{rank mismatch on result #0: expected rank 3}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<4x16xf16, 1>)
  return
}

// -----

// --- Positive: tensor mode (pre-bufferization) with all-static shapes. ---
// CHECK-LABEL: func.func @matmul_tensor_mode_static
// CHECK:         hip.matmul
func.func @matmul_tensor_mode_static(%ctx: !hip.context,
                                     %a: tensor<2x4xf16>,
                                     %b: tensor<4x8xf16>,
                                     %c: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
