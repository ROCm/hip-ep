// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// End-to-end pipeline test: bufferization -> pool-allocs -> alloc lowering.
//
// Models single-head scaled dot-product attention:
//   Q  = X @ Wq          K  = X @ Wk          V  = X @ Wv
//   KT = transpose(K)    scores = Q @ KT
//   scaled = scores * (1/sqrt(D))   probs = softmax(scaled)   out = probs @ V
//
// Allocation lifecycle through the pipeline:
//   tensor IR:          8 tensor.empty (no allocation yet)
//   after bufferize:    8 memref.alloc (each 2*64*64*4 = 32768 B)
//   after pool-allocs:  1 grow-on-demand pool sized to the concurrent peak of
//                       4 live 32768 B buffers = 131072 B, plus 8 memref.view.
//                       Disjoint-lifetime buffers share pool offsets, so the peak
//                       is 4 slots (the same reuse a liveness best-fit buffer
//                       pass would find) -- no separate reuse pass is needed.
//   after lower-allocs: unchanged -- every transient is a pool view and the pool
//                       is runtime-owned, so there are no hip.alloc / hip.free.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries" --hip-pool-allocs --hip-lower-allocs %s | FileCheck %s

// CHECK-LABEL: func.func @attention_pipeline

// One grow-on-demand pool sized to the 4-slot concurrent peak (4 * 32768 = 131072).
// CHECK:         %[[SZ:.*]] = arith.constant 131072 : index
// CHECK:         hip.get_pool(%arg0, %[[SZ]])

// The 8 transients become views into that single pool.
// CHECK-COUNT-8: memref.view

// Pooling subsumes buffer reuse: no standalone allocation survives, and the
// runtime-owned pool needs no per-buffer alloc/free.
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     hip.alloc
// CHECK-NOT:     hip.free

func.func @attention_pipeline(
    %ctx: !hip.context,
    %X: tensor<2x64x64xf32>,
    %Wq: tensor<64x64xf32>,
    %Wk: tensor<64x64xf32>,
    %Wv: tensor<64x64xf32>,
    %scale: tensor<f32>) -> tensor<2x64x64xf32> {
  // Q = X @ Wq  [B,S,D] @ [D,D] -> [B,S,D]
  %e0 = tensor.empty() : tensor<2x64x64xf32>
  %Q = hip.matmul(%ctx) ins(%X, %Wq : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e0 : tensor<2x64x64xf32>) : tensor<2x64x64xf32>

  // K = X @ Wk
  %e1 = tensor.empty() : tensor<2x64x64xf32>
  %K = hip.matmul(%ctx) ins(%X, %Wk : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e1 : tensor<2x64x64xf32>) : tensor<2x64x64xf32>

  // V = X @ Wv
  %e2 = tensor.empty() : tensor<2x64x64xf32>
  %V = hip.matmul(%ctx) ins(%X, %Wv : tensor<2x64x64xf32>, tensor<64x64xf32>) outs(%e2 : tensor<2x64x64xf32>) : tensor<2x64x64xf32>

  // KT = transpose(K, perm=[0,2,1])  [B,S,D] -> [B,D,S]
  %e3 = tensor.empty() : tensor<2x64x64xf32>
  %KT = hip.transpose(%ctx) ins(%K : tensor<2x64x64xf32>) outs(%e3 : tensor<2x64x64xf32>) {perm = [0, 2, 1]} : tensor<2x64x64xf32>

  // scores = Q @ KT  [B,S,D] @ [B,D,S] -> [B,S,S]
  %e4 = tensor.empty() : tensor<2x64x64xf32>
  %scores = hip.matmul(%ctx) ins(%Q, %KT : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%e4 : tensor<2x64x64xf32>) : tensor<2x64x64xf32>

  // scaled = scores * (1/sqrt(D))
  %e5 = tensor.empty() : tensor<2x64x64xf32>
  %scaled = hip.mul(%ctx) ins(%scores, %scale : tensor<2x64x64xf32>, tensor<f32>) outs(%e5 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  // probs = softmax(scaled, axis=-1)
  %e6 = tensor.empty() : tensor<2x64x64xf32>
  %probs = hip.miopen.softmax(%ctx) ins(%scaled : tensor<2x64x64xf32>) outs(%e6 : tensor<2x64x64xf32>) -> tensor<2x64x64xf32>

  // out = probs @ V  [B,S,S] @ [B,S,D] -> [B,S,D]
  %e7 = tensor.empty() : tensor<2x64x64xf32>
  %out = hip.matmul(%ctx) ins(%probs, %V : tensor<2x64x64xf32>, tensor<2x64x64xf32>) outs(%e7 : tensor<2x64x64xf32>) : tensor<2x64x64xf32>

  return %out : tensor<2x64x64xf32>
}
