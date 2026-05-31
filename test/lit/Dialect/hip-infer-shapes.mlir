// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-shapes %s | FileCheck %s

// What this file tests
// --------------------
// The `--hip-infer-shapes` production pass (lib/Dialect/Transforms/
// InferShapesPass.cpp) — i.e. the CONSUMER side of
// `ReifyRankedShapedTypeOpInterface` for HIP ops. Specifically:
//
//   - composing a refined result type by combining the current type
//     with the constant branches of each reified `OpFoldResult`
//     (`refine_single_matmul`, `refine_chained_matmul`),
//   - in-place SSA value-type narrowing and `tensor.empty` producer
//     rebuild when the producer is a single-use `tensor.empty`
//     (`refine_single_matmul`, `refine_chained_matmul`),
//   - leaving the producer alone when it is something else
//     (`skip_non_empty_producer`),
//   - early-out on already-static results (`noop_on_static`),
//   - filtering to ops in the `hip` dialect (`skip_non_hip_op`),
//   - inserting `tensor.cast` barriers on non-DPS uses while exempting
//     DPS-init uses so chains like
//     `matmul -> tensor.empty -> matmul` propagate refinement through
//     all links.
//
// What this file does NOT test
// ----------------------------
// The correctness of the `OpFoldResult`s produced by individual reify
// implementations — in particular, the dynamic-dim branch of
// `MatmulOp::reifyResultShapes` (which operand a dynamic dim is taken
// from, and at which local dim index). This pass only consumes the
// *constant* branch of each `OpFoldResult`; dynamic ones are silently
// discarded. That branch is covered by `hip-matmul-reify-shapes.mlir`,
// which uses upstream's `--resolve-shaped-type-result-dims` as a
// generic producer-contract validator (the upstream pass materializes
// every reified `OpFoldResult` — static and dynamic — into IR and is
// not itself part of our production pipeline).

// CHECK-LABEL: func.func @refine_single_matmul
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y:.*]] = hip.matmul
// CHECK-SAME:                  outs(%[[E]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[Y]] : tensor<2x8xf16> to tensor<?x?xf16>
// CHECK:         return %[[CAST]] : tensor<?x?xf16>
func.func @refine_single_matmul(%ctx: !hip.context,
                                %a: tensor<2x4xf16>,
                                %b: tensor<4x8xf16>,
                                %dM: index, %dN: index) -> tensor<?x?xf16> {
  %e = tensor.empty(%dM, %dN) : tensor<?x?xf16>
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @refine_chained_matmul
// CHECK:         %[[E1:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y1:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E1]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[Y1]] : tensor<2x8xf16> to tensor<?x?xf16>
// CHECK:         %[[E2:.*]] = tensor.empty() : tensor<2x16xf16>
// CHECK:         %[[Y2:.*]] = hip.matmul
// CHECK-SAME:      ins(%[[CAST]], %{{.*}} : tensor<?x?xf16>, tensor<8x16xf16>)
// CHECK-SAME:      outs(%[[E2]] : tensor<2x16xf16>) : tensor<2x16xf16>
func.func @refine_chained_matmul(%ctx: !hip.context,
                                 %a: tensor<2x4xf16>,
                                 %b: tensor<4x8xf16>,
                                 %c: tensor<8x16xf16>,
                                 %d1: index, %d2: index,
                                 %d3: index, %d4: index)
    -> tensor<?x?xf16> {
  %e1 = tensor.empty(%d1, %d2) : tensor<?x?xf16>
  %y1 = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e1 : tensor<?x?xf16>) : tensor<?x?xf16>
  %e2 = tensor.empty(%d3, %d4) : tensor<?x?xf16>
  %y2 = hip.matmul(%ctx)
    ins(%y1, %c : tensor<?x?xf16>, tensor<8x16xf16>)
    outs(%e2 : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y2 : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @skip_non_empty_producer
// CHECK:         hip.matmul
// CHECK-SAME:    outs(%{{.*}} : tensor<?x?xf16>) : tensor<?x?xf16>
func.func @skip_non_empty_producer(%ctx: !hip.context,
                                   %a: tensor<2x4xf16>,
                                   %b: tensor<4x8xf16>,
                                   %c: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func.func @noop_on_static
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x8xf16>
// CHECK:         %[[Y:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E]] : tensor<2x8xf16>) : tensor<2x8xf16>
// CHECK-NOT:    tensor.cast
// CHECK:         return %[[Y]] : tensor<2x8xf16>
func.func @noop_on_static(%ctx: !hip.context,
                          %a: tensor<2x4xf16>,
                          %b: tensor<4x8xf16>) -> tensor<2x8xf16> {
  %e = tensor.empty() : tensor<2x8xf16>
  %y = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e : tensor<2x8xf16>) : tensor<2x8xf16>
  return %y : tensor<2x8xf16>
}

// -----

// Pin the safety guard against refining a `tensor.empty` whose result is
// used as the outs operand of more than one HIP op. Refinement of one
// consumer's result would silently retype the shared empty (and thus
// every other consumer's outs operand) without retyping the others'
// results — breaking the DPS contract. No converter aliases empties
// today, so the guard is purely defensive against a future regression.
// CHECK-LABEL: func.func @skip_shared_empty_producer
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf16>
// CHECK:         %[[Y1:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK:         %[[Y2:.*]] = hip.matmul
// CHECK-SAME:      outs(%[[E]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK-NOT:     tensor.empty()
func.func @skip_shared_empty_producer(%ctx: !hip.context,
                                      %a: tensor<2x4xf16>,
                                      %b: tensor<4x8xf16>,
                                      %d1: index, %d2: index)
    -> (tensor<?x?xf16>, tensor<?x?xf16>) {
  %shared = tensor.empty(%d1, %d2) : tensor<?x?xf16>
  %y1 = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%shared : tensor<?x?xf16>) : tensor<?x?xf16>
  %y2 = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%shared : tensor<?x?xf16>) : tensor<?x?xf16>
  return %y1, %y2 : tensor<?x?xf16>, tensor<?x?xf16>
}

// -----

// Pin that the pass restricts itself to HIP-dialect ops: refining a non-HIP
// op like `tensor.pad` (which also implements ReifyRankedShapedTypeOpInterface)
// is the canonicalizer's job, not this pass's.
// CHECK-LABEL: func.func @skip_non_hip_op
// CHECK:         %[[P:.*]] = tensor.pad
// CHECK:         tensor<4x8xf16> to tensor<?x?xf16>
// CHECK-NOT:     tensor.cast
// CHECK:         return %[[P]] : tensor<?x?xf16>
func.func @skip_non_hip_op(%a: tensor<4x8xf16>) -> tensor<?x?xf16> {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f16
  %padded = tensor.pad %a low[%c1, %c1] high[%c1, %c1] {
    ^bb0(%i: index, %j: index):
      tensor.yield %cst : f16
  } : tensor<4x8xf16> to tensor<?x?xf16>
  return %padded : tensor<?x?xf16>
}

// -----

// `hip.rope` reify returns the input data tensor's shape: static dims
// transfer to the output (here, dim 2 = 4096); dynamic dims stay `?`.
// CHECK-LABEL: func.func @refine_rope_static_last_dim
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4096xf16>
// CHECK:         %[[Y:.*]] = hip.rope
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x4096xf16>) {{.*}} : tensor<?x?x4096xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x?x4096xf16> to tensor<?x?x?xf16>
func.func @refine_rope_static_last_dim(%ctx: !hip.context,
                                       %x: tensor<?x?x4096xf16>,
                                       %pos: tensor<?x?xi64>,
                                       %cos: tensor<131072x64xf16>,
                                       %sin: tensor<131072x64xf16>,
                                       %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf16> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = hip.rope(%ctx)
    ins(%x, %pos, %cos, %sin :
        tensor<?x?x4096xf16>, tensor<?x?xi64>,
        tensor<131072x64xf16>, tensor<131072x64xf16>)
    outs(%e : tensor<?x?x?xf16>)
    {interleaved = 0 : i64, num_heads = 32 : i64, rotary_embedding_dim = 128 : i64}
    : tensor<?x?x?xf16>
  return %y : tensor<?x?x?xf16>
}

// -----

// `hip.rms_norm` reify returns the input data tensor's shape; `$scale`
// broadcasts and does not contribute extents.
// CHECK-LABEL: func.func @refine_rms_norm_static_last_dim
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4096xf16>
// CHECK:         %[[Y:.*]] = hip.rms_norm
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x4096xf16>) {{.*}} : tensor<?x?x4096xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x?x4096xf16> to tensor<?x?x?xf16>
func.func @refine_rms_norm_static_last_dim(%ctx: !hip.context,
                                           %x: tensor<?x?x4096xf16>,
                                           %scale: tensor<4096xf16>,
                                           %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf16> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = hip.rms_norm(%ctx)
    ins(%x, %scale : tensor<?x?x4096xf16>, tensor<4096xf16>)
    outs(%e : tensor<?x?x?xf16>)
    {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    : tensor<?x?x?xf16>
  return %y : tensor<?x?x?xf16>
}

// -----

// `hip.qmoe` reify returns the input data tensor's shape — top-k expert
// routing happens inside the kernel and produces per-token outputs that
// are accumulated back into the original token slot.
// CHECK-LABEL: func.func @refine_qmoe_static_last_dim
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x2880xf16>
// CHECK:         %[[Y:.*]] = hip.qmoe
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x2880xf16>) {{.*}} : tensor<?x?x2880xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x?x2880xf16> to tensor<?x?x?xf16>
func.func @refine_qmoe_static_last_dim(
    %ctx: !hip.context,
    %x: tensor<?x?x2880xf16>,
    %router: tensor<?x32xf16>,
    %fc1w: tensor<32x5760x1440xui8>,
    %fc1s: tensor<32x5760x90xf16>,
    %fc2w: tensor<32x2880x1440xui8>,
    %fc2s: tensor<32x2880x90xf16>,
    %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf16> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = hip.qmoe(%ctx)
    ins(%x, %router, %fc1w, %fc1s, %fc2w, %fc2s :
        tensor<?x?x2880xf16>, tensor<?x32xf16>,
        tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>,
        tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>)
    outs(%e : tensor<?x?x?xf16>)
    {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
     normalize_routing_weights = 0 : i64, swiglu_fusion = 1 : i64,
     use_sparse_mixer = 0 : i64,
     activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32,
     swiglu_limit = 0x7F800000 : f32, activation_type = "swiglu"}
    : tensor<?x?x?xf16>
  return %y : tensor<?x?x?xf16>
}

// -----

// `hip.matmul_nbits` reify takes leading dims from `$A` and the final
// dim from the integer attribute `$N` (always static).
// CHECK-LABEL: func.func @refine_matmul_nbits_from_attr
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x5120xf16>
// CHECK:         %[[Y:.*]] = hip.matmul_nbits
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x5120xf16>) {{.*}} : tensor<?x?x5120xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x?x5120xf16> to tensor<?x?x?xf16>
func.func @refine_matmul_nbits_from_attr(%ctx: !hip.context,
                                         %a: tensor<?x?x2880xf16>,
                                         %b: tensor<5120x90x16xui8>,
                                         %scales: tensor<5120x90xf16>,
                                         %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf16> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = hip.matmul_nbits(%ctx)
    ins(%a, %b, %scales :
        tensor<?x?x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>)
    outs(%e : tensor<?x?x?xf16>)
    {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
     block_size = 32 : i64, accuracy_level = 4 : i64, zp_elem_size = 1 : i64}
    : tensor<?x?x?xf16>
  return %y : tensor<?x?x?xf16>
}

// -----

// `hip.gemm` reify produces 2D `[M, N]`:
//   M = transA ? A.shape[1] : A.shape[0]
//   N = transB ? B.shape[0] : B.shape[1]
// Static M from A and static N from B propagate to the result.
// `hip.gemm`'s alpha/beta/transA/transB are DefaultValuedAttrs (1.0, 1.0,
// 0, 0); when all four are at default the pretty-printer emits no
// attr-dict between `outs(...)` and the trailing result-type spec — hence
// no `{{.*}}` placeholder between them in the CHECK below.
// CHECK-LABEL: func.func @refine_gemm_2d_from_inputs
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<128x256xf32>
// CHECK:         %[[Y:.*]] = hip.gemm
// CHECK-SAME:                  outs(%[[E]] : tensor<128x256xf32>) : tensor<128x256xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<128x256xf32> to tensor<?x?xf32>
func.func @refine_gemm_2d_from_inputs(%ctx: !hip.context,
                                      %a: tensor<128x?xf32>,
                                      %b: tensor<?x256xf32>,
                                      %dM: index, %dN: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%dM, %dN) : tensor<?x?xf32>
  %y = hip.gemm(%ctx)
    ins(%a, %b : tensor<128x?xf32>, tensor<?x256xf32>)
    outs(%e : tensor<?x?xf32>)
    {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : i64, transB = 0 : i64}
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// Same-shape unary ops (silu, sigmoid, softplus, gelu, reciprocal, sqrt,
// not, cos, sin, neg, cast, sign, cumsum, scatter_nd) all reify their
// result shape from the primary input tensor's shape — covered here by
// `hip.cos` as a representative. The pass refines the dynamic-result
// `tensor.empty` to the input's static rank/extents.
// CHECK-LABEL: func.func @refine_cos_static_last_dim
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x4096xf32>
// CHECK:         %[[Y:.*]] = hip.cos
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x4096xf32>) : tensor<?x?x4096xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<?x?x4096xf32> to tensor<?x?x?xf32>
func.func @refine_cos_static_last_dim(%ctx: !hip.context,
                                      %x: tensor<?x?x4096xf32>,
                                      %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf32> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %y = hip.cos(%ctx)
    ins(%x : tensor<?x?x4096xf32>)
    outs(%e : tensor<?x?x?xf32>)
    : tensor<?x?x?xf32>
  return %y : tensor<?x?x?xf32>
}

// -----

// `hip.size` produces a rank-0 `tensor<i64>` — its reify returns a
// shape entry with an empty inner dim list. The pass has nothing to
// refine (rank-0 is already maximally static), but must not crash on
// the empty inner list. CHECK that the op survives the pass unchanged
// and no `tensor.cast` barrier is inserted (the result type is already
// the same as the function's return type).
// CHECK-LABEL: func.func @noop_on_size_rank_zero
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<i64>
// CHECK:         %[[Y:.*]] = hip.size
// CHECK-SAME:                  outs(%[[E]] : tensor<i64>) : tensor<i64>
// CHECK-NOT:     tensor.cast
// CHECK:         return %[[Y]] : tensor<i64>
func.func @noop_on_size_rank_zero(%ctx: !hip.context,
                                  %x: tensor<?x?xf16>) -> tensor<i64> {
  %e = tensor.empty() : tensor<i64>
  %y = hip.size(%ctx)
    ins(%x : tensor<?x?xf16>)
    outs(%e : tensor<i64>)
    : tensor<i64>
  return %y : tensor<i64>
}
