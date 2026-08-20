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
// PR #265 commit 1 added Phase 2 (`refineLoopSignatures`):
//
//   - idempotence on already-tight `hip.loop` signatures
//     (`loop_signatures_no_op`),
//   - depth-agnostic refinement across nested `hip.loop` ops via
//     bounded fixed-point iteration (`nested_loop_signatures`).
//
// Production-pipeline coverage of `hip.loop` signature refinement
// driven by an upstream `hip.matmul` narrowing — the case the helper
// was actually written to fix — lands together with PR #265 commit 2
// once `OnnxLoopOutlinePass` flips its body-arg + result-type
// producers to read v_init / inferReturnTypes.
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
// CHECK:         %[[CAST:.*]] = tensor.cast %[[Y1]] : tensor<2x8xf16> to tensor<?x8xf16>
// CHECK:         %[[E2:.*]] = tensor.empty() : tensor<2x16xf16>
// CHECK:         %[[Y2:.*]] = hip.matmul
// CHECK-SAME:      ins(%[[CAST]], %{{.*}} : tensor<?x8xf16>, tensor<8x16xf16>)
// CHECK-SAME:      outs(%[[E2]] : tensor<2x16xf16>) : tensor<2x16xf16>
func.func @refine_chained_matmul(%ctx: !hip.context,
                                 %a: tensor<2x4xf16>,
                                 %b: tensor<4x8xf16>,
                                 %c: tensor<8x16xf16>,
                                 %d1: index, %d2: index,
                                 %d3: index, %d4: index)
    -> tensor<?x?xf16> {
  %e1 = tensor.empty(%d1) : tensor<?x8xf16>
  %y1 = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%e1 : tensor<?x8xf16>) : tensor<?x8xf16>
  %e2 = tensor.empty(%d3, %d4) : tensor<?x?xf16>
  %y2 = hip.matmul(%ctx)
    ins(%y1, %c : tensor<?x8xf16>, tensor<8x16xf16>)
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
                                      %a: tensor<128x64xf32>,
                                      %b: tensor<64x256xf32>,
                                      %dM: index, %dN: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%dM, %dN) : tensor<?x?xf32>
  %y = hip.gemm(%ctx)
    ins(%a, %b : tensor<128x64xf32>, tensor<64x256xf32>)
    outs(%e : tensor<?x?xf32>)
    {alpha = 1.0 : f32, beta = 1.0 : f32,
     transA = 0 : i64, transB = 0 : i64}
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// Same-shape unary ops (silu, sigmoid, softplus, gelu, reciprocal, sqrt,
// not, cos, sin, neg, cast, sign, cumsum, scatter_nd) select the shared
// `HipDpsOp` default-reify family, which lifts each
// output dim from the DPS `outs` operand's runtime shape. The DPS
// contract pins `result.type == outs.type`, so the consumer-side pass
// has signal to refine only when `outs` carries static dims that the
// result type doesn't (rare in practice; the in-tree converters keep
// `outs.type` aligned with the inferred ONNX result type from the
// start). When `outs` is fully dynamic, the pass becomes a no-op for
// these ops — `hip.cos` is the canonical example pinned here.
// CHECK-LABEL: func.func @noop_on_cos_dynamic_outs
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
// CHECK:         %[[Y:.*]] = hip.cos
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?x?xf32>) : tensor<?x?x?xf32>
// CHECK-NOT:     tensor.cast
// CHECK:         return %[[Y]] : tensor<?x?x?xf32>
func.func @noop_on_cos_dynamic_outs(%ctx: !hip.context,
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

// -----

// 2-operand broadcast ops (miopen.add, mul, add, min, div, equal, and,
// sub, less, mod) all reify their result shape via NumPy broadcast over
// both inputs — covered here by `hip.add` as a representative. The
// canonical-side pick is exercised: lhs=<?x4> contributes the static `4`
// at dim 1, rhs=<2x?> contributes the static `2` at dim 0, so the
// fully-dynamic result type tightens to `tensor<2x4xf16>`.
//
// Note: `hip.add` uses a custom assembly format (alongside `hip.mul`
// and `hip.miopen.add`) that emits the result type with `-> type`, not
// `: type` — the latter is the TableGen `assemblyFormat` convention
// used by the other commit-2 broadcast ops (min, div, equal, and, sub,
// where, less, mod) and by `hip.matmul`.
// CHECK-LABEL: func.func @refine_add_broadcast_canonical_pick
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x4xf16>
// CHECK:         %[[Y:.*]] = hip.add
// CHECK-SAME:                  outs(%[[E]] : tensor<2x4xf16>) -> tensor<2x4xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<2x4xf16> to tensor<?x?xf16>
func.func @refine_add_broadcast_canonical_pick(%ctx: !hip.context,
                                                %lhs: tensor<?x4xf16>,
                                                %rhs: tensor<2x?xf16>,
                                                %d0: index, %d1: index)
    -> tensor<?x?xf16> {
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %y = hip.add(%ctx)
    ins(%lhs, %rhs : tensor<?x4xf16>, tensor<2x?xf16>)
    outs(%e : tensor<?x?xf16>)
    -> tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// 3-operand `hip.where` reifies its result shape via NumPy broadcast over
// `condition`, `x`, `y`. With cond=<1x4xi1>, x=<2x1xf32>, y=<2x4xf32>,
// the broadcast result is `tensor<2x4xf32>` — `where` is the only
// commit-2 op with a non-binary input list, so it gets its own LIT case
// to guard the variadic helper path. `hip.where` uses TableGen
// `assemblyFormat` so the trailing result type is printed with `:`.
// CHECK-LABEL: func.func @refine_where_3operand_broadcast
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK:         %[[Y:.*]] = hip.where
// CHECK-SAME:                  outs(%[[E]] : tensor<2x4xf32>) : tensor<2x4xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<2x4xf32> to tensor<?x?xf32>
func.func @refine_where_3operand_broadcast(%ctx: !hip.context,
                                            %cond: tensor<1x4xi1>,
                                            %x: tensor<2x1xf32>,
                                            %y: tensor<2x4xf32>,
                                            %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %out = hip.where(%ctx)
    ins(%cond, %x, %y : tensor<1x4xi1>, tensor<2x1xf32>, tensor<2x4xf32>)
    outs(%e : tensor<?x?xf32>)
    : tensor<?x?xf32>
  return %out : tensor<?x?xf32>
}

// -----

// `hip.transpose` (PR #263 commit 3) reifies its result shape by
// permuting the input shape via the static `perm` attribute — output
// dim i comes from input dim `perm[i]`. With input=<2x?x4096xf16> and
// perm=[2, 0, 1], the result tightens from fully-dynamic
// `tensor<?x?x?xf16>` to `tensor<4096x2x?xf16>` (dim 0 = input dim 2 =
// 4096, dim 1 = input dim 0 = 2; dim 2 = input dim 1 stays dynamic).
// CHECK-LABEL: func.func @refine_transpose_perm_driven
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<4096x2x?xf16>
// CHECK:         %[[Y:.*]] = hip.transpose
// CHECK-SAME:                  outs(%[[E]] : tensor<4096x2x?xf16>){{.*}}: tensor<4096x2x?xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<4096x2x?xf16> to tensor<?x?x?xf16>
func.func @refine_transpose_perm_driven(%ctx: !hip.context,
                                        %x: tensor<2x?x4096xf16>,
                                        %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf16> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %y = hip.transpose(%ctx)
    ins(%x : tensor<2x?x4096xf16>)
    outs(%e : tensor<?x?x?xf16>)
    {perm = [2, 0, 1]}
    : tensor<?x?x?xf16>
  return %y : tensor<?x?x?xf16>
}

// -----

// `hip.gather` (PR #263 commit 3) reifies output as
// `data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`. With
// data=<?x4xf32>, indices=<2x?xi64>, axis=0: output dims =
// [indices[0]=2, indices[1]=?, data[1]=4] = `tensor<2x?x4xf32>`. The
// pass tightens dim 0 (static 2 from indices) and dim 2 (static 4 from
// data); dim 1 stays dynamic.
// CHECK-LABEL: func.func @refine_gather_axis_split
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<2x?x4xf32>
// CHECK:         %[[Y:.*]] = hip.gather
// CHECK-SAME:                  outs(%[[E]] : tensor<2x?x4xf32>){{.*}}: tensor<2x?x4xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<2x?x4xf32> to tensor<?x?x?xf32>
func.func @refine_gather_axis_split(%ctx: !hip.context,
                                    %data: tensor<?x4xf32>,
                                    %indices: tensor<2x?xi64>,
                                    %d0: index, %d1: index, %d2: index)
    -> tensor<?x?x?xf32> {
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %y = hip.gather(%ctx)
    ins(%data, %indices : tensor<?x4xf32>, tensor<2x?xi64>)
    outs(%e : tensor<?x?x?xf32>)
    {axis = 0 : i64}
    : tensor<?x?x?xf32>
  return %y : tensor<?x?x?xf32>
}

// -----

// `hip.gather_nd` (PR #263 commit 3) reifies output as
// `data.shape[:batch_dims] ++ indices.shape[batch_dims:-1] ++
//  data.shape[batch_dims + indices.shape[-1]:]`. With data=<3x4x5xf32>,
// indices=<2x2xi64> (tupleWidth=2), batch_dims=0: output rank =
// 2 + 3 - 2 - 1 - 0 = 2; output = [] ++ [indices[0]=2] ++ [data[2]=5] =
// `tensor<2x5xf32>`.
// CHECK-LABEL: func.func @refine_gather_nd_structural
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x5xf32>
// CHECK:         %[[Y:.*]] = hip.gather_nd
// CHECK-SAME:                  outs(%[[E]] : tensor<2x5xf32>){{.*}}: tensor<2x5xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<2x5xf32> to tensor<?x?xf32>
func.func @refine_gather_nd_structural(%ctx: !hip.context,
                                       %data: tensor<3x4x5xf32>,
                                       %indices: tensor<2x2xi64>,
                                       %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = hip.gather_nd(%ctx)
    ins(%data, %indices : tensor<3x4x5xf32>, tensor<2x2xi64>)
    outs(%e : tensor<?x?xf32>)
    {batch_dims = 0 : i64}
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// `hip.reduce_sum` (PR #263 commit 3) reifies output by introspecting
// the `axes` operand as an `arith.constant`. With data=<?x4096xf16>,
// axes=dense<[1]>, keepdims=1, noop_with_empty_axes=0: dim 0 passes
// through from data (dynamic — emits `tensor.dim %data`), dim 1 is the
// reduced axis with keepdims=1 → static 1. The pass tightens dim 1
// only; dim 0 stays dynamic. Same path covers `hip.reduce_max` and
// `hip.reduce_prod` (one helper, three thunks). When `axes` is not a
// constant the helper bails and the op falls through to a no-op
// outs-shape fallback — verified by `noop_on_static` earlier in this
// file, since the fallback only emits dim ops that the pass discards.
// CHECK-LABEL: func.func @refine_reduce_sum_keepdims_constant_axes
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<?x1xf16>
// CHECK:         %[[Y:.*]] = hip.reduce_sum
// CHECK-SAME:                  outs(%[[E]] : tensor<?x1xf16>){{.*}}: tensor<?x1xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x1xf16> to tensor<?x?xf16>
func.func @refine_reduce_sum_keepdims_constant_axes(%ctx: !hip.context,
                                                    %data: tensor<?x4096xf16>,
                                                    %d0: index, %d1: index)
    -> tensor<?x?xf16> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %y = hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<?x4096xf16>, tensor<1xi64>)
    outs(%e : tensor<?x?xf16>)
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// `hip.reduce_mean` shares the `Hip_DpsOp_Reduction` reify/InferType path with
// reduce_sum/max/prod (same `reifyReductionShape` helper), so the constant-axes
// refinement applies identically: data=<?x4096xf16>, axes=dense<[1]>,
// keepdims=1 → dim 0 passes through (dynamic), dim 1 tightens to static 1.
// This is the key scalability property: dynamic Reshape result dims that an
// ONNX-level result-type refiner would once have recovered pre-conversion are
// now recovered post-conversion by this dialect-level pass instead.
// CHECK-LABEL: func.func @refine_reduce_mean_keepdims_constant_axes
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<?x1xf16>
// CHECK:         %[[Y:.*]] = hip.reduce_mean
// CHECK-SAME:                  outs(%[[E]] : tensor<?x1xf16>){{.*}}: tensor<?x1xf16>
// CHECK:         tensor.cast %[[Y]] : tensor<?x1xf16> to tensor<?x?xf16>
func.func @refine_reduce_mean_keepdims_constant_axes(%ctx: !hip.context,
                                                     %data: tensor<?x4096xf16>,
                                                     %d0: index, %d1: index)
    -> tensor<?x?xf16> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %y = hip.reduce_mean(%ctx)
    ins(%data, %axes : tensor<?x4096xf16>, tensor<1xi64>)
    outs(%e : tensor<?x?xf16>)
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<?x?xf16>
  return %y : tensor<?x?xf16>
}

// -----

// `hip.pad` (along with tile, expand, slice, range) uses the shared
// default-reify family — the generated dispatcher walks
// `getDpsInits()` and lifts each output dim from the DPS `outs`
// operand's runtime shape via `tensor::getMixedSizes`. Static dims
// become `IndexAttr` (which the pass can use to tighten); dynamic
// dims emit `tensor.dim %output, %i` (a no-op tightening that the
// pass discards). Per-op arithmetic for these ops (e.g.
// `tensor::PadOp`'s `affine.apply (d0 + d1 + d2)` recipe) is deferred.
//
// This LIT case is the contract guard for the default outs-lift path:
//   1. The op's reify ALWAYS returns success() — no crash, no failure
//      propagation up to the pass.
//   2. Fully-dynamic `outs` survives the pass intact: the pass walks
//      the lifted `tensor.dim` outputs, finds nothing it can use to
//      tighten, and leaves the op unchanged.
//
// We also exercise the optional `cval` branch of pad's assembly format
// here (the most common ONNX Pad shape) — the reify path doesn't read
// `cval` but we want to make sure the auto-emit works on the variant
// with a non-empty optional segment, not just the bare-minimum shape.
//
// (Same default path also covers tile, expand, slice, range — one LIT
// case is enough to guard the contract.)
// CHECK-LABEL: func.func @refine_pad_dps_out_fallback
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
// CHECK:         %[[Y:.*]] = hip.pad
// CHECK-SAME:                  outs(%[[E]] : tensor<?x?xf32>){{.*}}: tensor<?x?xf32>
// CHECK:         return %[[Y]] : tensor<?x?xf32>
func.func @refine_pad_dps_out_fallback(%ctx: !hip.context,
                                       %data: tensor<3x2xf32>,
                                       %pads: tensor<4xi64>,
                                       %cval: tensor<f32>,
                                       %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = hip.pad(%ctx)
    ins(%data, %pads : tensor<3x2xf32>, tensor<4xi64>)
    cval(%cval : tensor<f32>)
    outs(%e : tensor<?x?xf32>)
    {mode = "constant"}
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// `hip.pad` Tier-1 promotion. With static `data` and constant `pads`,
// the helper computes
// `output[d] = data.shape[d] + pads[d] + pads[d + N]`. Pads = [1, 2, 1, 2]
// over axes [0, 1] yields output [3+1+1=5, 4+2+2=8]. The pass refines
// the result type from `tensor<?x?xf32>` to `tensor<5x8xf32>` and the
// outs `tensor.empty` producer is rebuilt without dynamic operands.
// CHECK-LABEL: func.func @refine_pad_tier1_constant_pads
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<5x8xf32>
// CHECK:         %[[Y:.*]] = hip.pad
// CHECK-SAME:                  outs(%[[E]] : tensor<5x8xf32>){{.*}}: tensor<5x8xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<5x8xf32> to tensor<?x?xf32>
func.func @refine_pad_tier1_constant_pads(%ctx: !hip.context,
                                          %data: tensor<3x4xf32>,
                                          %cval: tensor<f32>,
                                          %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %pads = arith.constant dense<[1, 2, 1, 2]> : tensor<4xi64>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = hip.pad(%ctx)
    ins(%data, %pads : tensor<3x4xf32>, tensor<4xi64>)
    cval(%cval : tensor<f32>)
    outs(%e : tensor<?x?xf32>)
    {mode = "constant"}
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// `hip.tile` Tier-1 promotion. With static `input` and constant
// `repeats`, the helper computes
// `output[d] = input.shape[d] * repeats[d]`. Repeats = [2, 3] over a
// `tensor<3x4xf32>` yields output [6, 12].
// CHECK-LABEL: func.func @refine_tile_tier1_constant_repeats
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<6x12xf32>
// CHECK:         %[[Y:.*]] = hip.tile
// CHECK-SAME:                  outs(%[[E]] : tensor<6x12xf32>){{.*}}: tensor<6x12xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<6x12xf32> to tensor<?x?xf32>
func.func @refine_tile_tier1_constant_repeats(%ctx: !hip.context,
                                               %input: tensor<3x4xf32>,
                                               %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = hip.tile(%ctx)
    ins(%input, %repeats : tensor<3x4xf32>, tensor<2xi64>)
    outs(%e : tensor<?x?xf32>)
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// `hip.expand` Tier-1 promotion. With static `input` and constant
// `shape`, the helper runs the standard NumPy broadcast (right-
// aligned, leading-1 padded). Input `tensor<3x1xf32>` broadcast
// against shape [2, 3, 6] yields output [2, 3, 6].
// CHECK-LABEL: func.func @refine_expand_tier1_constant_shape
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<2x3x6xf32>
// CHECK:         %[[Y:.*]] = hip.expand
// CHECK-SAME:                  outs(%[[E]] : tensor<2x3x6xf32>){{.*}}: tensor<2x3x6xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<2x3x6xf32> to tensor<?x?x?xf32>
func.func @refine_expand_tier1_constant_shape(%ctx: !hip.context,
                                                %input: tensor<3x1xf32>,
                                                %d0: index, %d1: index,
                                                %d2: index)
    -> tensor<?x?x?xf32> {
  %shape = arith.constant dense<[2, 3, 6]> : tensor<3xi64>
  %e = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %valid = arith.constant true
  %y = hip.expand(%ctx) valid(%valid)
    ins(%input, %shape : tensor<3x1xf32>, tensor<3xi64>)
    outs(%e : tensor<?x?x?xf32>)
    : tensor<?x?x?xf32>
  return %y : tensor<?x?x?xf32>
}

// -----

// `hip.slice` Tier-1 promotion. With static `data` and constant
// `starts` / `ends` (axes default to [0..rank), steps default to
// all-ones), the helper computes
// `output[axis] = ceil_div(end - start, step)`. Slicing `tensor<10x4xf32>`
// with starts=[1, 0], ends=[6, 4] yields output [5, 4].
// CHECK-LABEL: func.func @refine_slice_tier1_constant_indices
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<5x4xf32>
// CHECK:         %[[Y:.*]] = hip.slice
// CHECK-SAME:                  outs(%[[E]] : tensor<5x4xf32>){{.*}}: tensor<5x4xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<5x4xf32> to tensor<?x?xf32>
func.func @refine_slice_tier1_constant_indices(%ctx: !hip.context,
                                                 %data: tensor<10x4xf32>,
                                                 %d0: index, %d1: index)
    -> tensor<?x?xf32> {
  %starts = arith.constant dense<[1, 0]> : tensor<2xi64>
  %ends = arith.constant dense<[6, 4]> : tensor<2xi64>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %y = hip.slice(%ctx)
    ins(%data, %starts, %ends : tensor<10x4xf32>, tensor<2xi64>, tensor<2xi64>)
    outs(%e : tensor<?x?xf32>)
    : tensor<?x?xf32>
  return %y : tensor<?x?xf32>
}

// -----

// `hip.slice` non-foldable case: `starts` is a function arg, not a
// constant. The Tier-1 helper bails and the per-op thunk falls back
// to `HipDpsOp::reifyResultShapes`'s outs-lift default. Static dims
// of the outs (`tensor<10x?xf32>`) still tighten dim 0; the dynamic
// dim 1 stays dynamic. This guards "no IR bloat on non-foldable slice"
// (the dim-arith chain would have been
// `arith.divsi(arith.subi(end, start), step)` per axis -- never emitted).
// CHECK-LABEL: func.func @refine_slice_non_foldable_falls_back_to_outs
// CHECK-NOT:     arith.subi
// CHECK-NOT:     arith.divsi
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<10x?xf32>
// CHECK:         %[[Y:.*]] = hip.slice
// CHECK-SAME:                  outs(%[[E]] : tensor<10x?xf32>){{.*}}: tensor<10x?xf32>
// CHECK:         tensor.cast %[[Y]] : tensor<10x?xf32> to tensor<?x?xf32>
func.func @refine_slice_non_foldable_falls_back_to_outs(
    %ctx: !hip.context,
    %data: tensor<10x4xf32>,
    %starts: tensor<2xi64>,
    %ends: tensor<2xi64>,
    %d0: index)
    -> tensor<?x?xf32> {
  %e = tensor.empty(%d0) : tensor<10x?xf32>
  %y = hip.slice(%ctx)
    ins(%data, %starts, %ends : tensor<10x4xf32>, tensor<2xi64>, tensor<2xi64>)
    outs(%e : tensor<10x?xf32>)
    : tensor<10x?xf32>
  %c = tensor.cast %y : tensor<10x?xf32> to tensor<?x?xf32>
  return %c : tensor<?x?xf32>
}

// -----

// `hip.range` Tier-1 promotion. With constant rank-0 start / limit /
// delta, the helper computes
// `count = ceil_div(max(0, limit - start), delta)`. start=2, limit=10,
// delta=3 yields count=3.
// CHECK-LABEL: func.func @refine_range_tier1_constant_args
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<3xi64>
// CHECK:         %[[Y:.*]] = hip.range
// CHECK-SAME:                  outs(%[[E]] : tensor<3xi64>){{.*}}: tensor<3xi64>
// CHECK:         tensor.cast %[[Y]] : tensor<3xi64> to tensor<?xi64>
func.func @refine_range_tier1_constant_args(%ctx: !hip.context,
                                              %d0: index)
    -> tensor<?xi64> {
  %start = arith.constant dense<2> : tensor<i64>
  %limit = arith.constant dense<10> : tensor<i64>
  %delta = arith.constant dense<3> : tensor<i64>
  %e = tensor.empty(%d0) : tensor<?xi64>
  %y = hip.range(%ctx)
    ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
    outs(%e : tensor<?xi64>)
    : tensor<?xi64>
  return %y : tensor<?xi64>
}

// -----

// `hip.gqa` is multi-result (3 required outs: output, present_key,
// present_value, plus one optional debug out `output_qk` -- omitted
// here). Reify routes through `Hip_DpsOp`'s default `reifyResultShapes`
// body in `lib/Dialect/IR/HipDpsOpInterface.cpp`, which walks
// `getDpsInits()` and lifts each init's runtime shape via
// `tensor::getMixedSizes` -- one entry of `reified` per result. The
// LIT case uses matching dynamic outs/result triples and guards the
// no-crash + multi-result-walk contract; for static-dim refinement
// see the per-op-arithmetic cases like `refine_transpose_perm_driven`.
//
// Pass behavior: walks all 3 results, calls reify which lifts each
// outs's `tensor<?x?x?xf16>` / `tensor<?x?x?x?xf16>` shape as
// `tensor.dim` ops, finds nothing static-er to narrow to, and exits
// without mutating the op. The CHECK guards that the op stays intact
// across all three outs and the dim-walk doesn't bail mid-result.
// CHECK-LABEL: func.func @refine_gqa_multi_result_dps_out_fallback
// CHECK:         %[[E0:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf16>
// CHECK:         %[[E1:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?x?xf16>
// CHECK:         %[[E2:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?x?xf16>
// CHECK:         %[[R:.*]]:3 = hip.gqa
// CHECK-SAME:      outs(%[[E0]], %[[E1]], %[[E2]] :
// CHECK-SAME:           tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
// CHECK:         return %[[R]]#0, %[[R]]#1, %[[R]]#2
func.func @refine_gqa_multi_result_dps_out_fallback(
    %ctx: !hip.context,
    %query: tensor<1x?x4096xf16>,
    %key: tensor<1x?x1024xf16>,
    %value: tensor<1x?x1024xf16>,
    %past_key: tensor<1x8x?x128xf16>,
    %past_value: tensor<1x8x?x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>,
    %d0: index, %d1: index, %d2: index, %d3: index)
    -> (tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) {
  %e0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %e1 = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf16>
  %e2 = tensor.empty(%d0, %d1, %d2, %d3) : tensor<?x?x?x?xf16>
  %r:3 = hip.gqa(%ctx)
    ins(%query, %key, %value, %past_key, %past_value,
        %seqlens_k, %total_seq_len :
        tensor<1x?x4096xf16>, tensor<1x?x1024xf16>, tensor<1x?x1024xf16>,
        tensor<1x8x?x128xf16>, tensor<1x8x?x128xf16>,
        tensor<1xi32>, tensor<i32>)
    outs(%e0, %e1, %e2 :
         tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
    {num_heads = 32 : i64, kv_num_heads = 8 : i64,
     scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32,
     do_rotary = 0 : i64, rotary_interleaved = 0 : i64,
     local_window_size = -1 : i64}
    : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  return %r#0, %r#1, %r#2 : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
}

// -----

// `hip.layer_norm` is variadic-multi-result: 1, 2, or 3 outs (output
// required; mean and inv_std optional, training-only). The default
// `Hip_DpsOp::reifyResultShapes` body walks `getDpsInits()` regardless
// of arity; this case exercises the full 3-out form so the variadic
// iteration is covered (1-out covered implicitly by every Tier-2
// fallback test).
//
// Same matching-dynamic-outs pattern as the gqa case above: pass walks
// all 3 results, reify lifts dynamic outs, no narrowing happens, op
// stays intact -- guards "Variadic outputs interpreted correctly".
// CHECK-LABEL: func.func @refine_layer_norm_variadic_three_outs
// CHECK:         %[[E0:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf16>
// CHECK:         %[[E1:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
// CHECK:         %[[E2:.*]] = tensor.empty(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<?x?x?xf32>
// CHECK:         %[[R:.*]]:3 = hip.layer_norm
// CHECK-SAME:      outs(%[[E0]], %[[E1]], %[[E2]] :
// CHECK-SAME:           tensor<?x?x?xf16>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK:         return %[[R]]#0, %[[R]]#1, %[[R]]#2
func.func @refine_layer_norm_variadic_three_outs(
    %ctx: !hip.context,
    %input: tensor<?x?x?xf16>,
    %scale: tensor<?xf16>,
    %d0: index, %d1: index, %d2: index)
    -> (tensor<?x?x?xf16>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %e0 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf16>
  %e1 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %e2 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32>
  %r:3 = hip.layer_norm(%ctx)
    ins(%input, %scale : tensor<?x?x?xf16>, tensor<?xf16>)
    outs(%e0, %e1, %e2 :
         tensor<?x?x?xf16>, tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    : tensor<?x?x?xf16>, tensor<?x?x?xf32>, tensor<?x?x?xf32>
  return %r#0, %r#1, %r#2 : tensor<?x?x?xf16>, tensor<?x?x?xf32>, tensor<?x?x?xf32>
}

// -----

// `hip.nonzero` has a data-dependent output dim 1 (number of non-zero
// elements -- only knowable at runtime). The default DPS Tier-2 reify
// lifts the outs's shape directly: dim 0 is the static input rank,
// dim 1 lifts as `tensor.dim %outs, %c1` and stays dynamic because
// outs itself was constructed dynamic. This is the "outs IS the only
// authority on the output shape" case the Tier-2 contract is designed
// for -- the LIT case guards that reify doesn't crash on data-
// dependent shapes.
// CHECK-LABEL: func.func @refine_nonzero_data_dependent_dim1
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xi64>
// CHECK:         %[[C:.*]] = tensor.empty() : tensor<i32>
// CHECK:         %[[Y:.*]]:2 = hip.nonzero
// CHECK-SAME:      outs(%[[E]], %[[C]] : tensor<?x?xi64>, tensor<i32>){{.*}}: tensor<?x?xi64>, tensor<i32>
// CHECK:         return %[[Y]]#0 : tensor<?x?xi64>
func.func @refine_nonzero_data_dependent_dim1(
    %ctx: !hip.context,
    %x: tensor<3x4xf16>,
    %d0: index, %d1: index)
    -> tensor<?x?xi64> {
  %e = tensor.empty(%d0, %d1) : tensor<?x?xi64>
  %c = tensor.empty() : tensor<i32>
  %y, %cnt = hip.nonzero(%ctx) ins(%x : tensor<3x4xf16>)
                          outs(%e, %c : tensor<?x?xi64>, tensor<i32>)
                          {input_data_type = 1 : i64}
                          : tensor<?x?xi64>, tensor<i32>
  return %y : tensor<?x?xi64>
}

// -----

// PR #265 commit 1 — Phase 2 (`refineLoopSignatures`) idempotence guard.
//
// `hip.loop` v_init type already matches the body func's declared arg
// type at slot [3..3+N) and the loop's own result type. Phase 2 must
// be a no-op: no signature mutation, no body re-walk, no extra
// `tensor.cast` insertion. The body func's `func.return` SSA wiring
// is also already type-consistent.
//
// CHECK-LABEL: func.func private @body_already_tight
// CHECK-SAME:    %[[V:[^ ,]+]]: tensor<128xf32>
// CHECK-SAME:    -> (i32, tensor<128xf32>)
// CHECK:         return %{{.*}}, %[[V]] : i32, tensor<128xf32>
func.func private @body_already_tight(%ctx: !hip.context, %iter: tensor<i64>,
                                      %cond_in: tensor<i1>,
                                      %v: tensor<128xf32>,
                                      %frame: !hip.loop_frame) -> (i32, tensor<128xf32>) {
  %status = arith.constant 0 : i32
  return %status, %v : i32, tensor<128xf32>
}

// CHECK-LABEL: func.func @loop_signatures_no_op
// CHECK-NOT:     tensor.cast
// CHECK:         %[[R:.*]] = hip.loop
// CHECK-SAME:      iter_args(%{{.*}} : tensor<128xf32>)
// CHECK-SAME:      -> (tensor<128xf32>)
// CHECK-SAME:      body @body_already_tight
// CHECK:         return %[[R]] : tensor<128xf32>
func.func @loop_signatures_no_op(%ctx: !hip.context,
                                 %M: index, %cond: i1,
                                 %v_static: tensor<128xf32>) -> tensor<128xf32> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%v_static : tensor<128xf32>)
                 -> (tensor<128xf32>)
                 body @body_already_tight
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<128xf32>
}

// -----

// PR #265 commit 1 — Phase 2 depth-agnosticism guard.
//
// Outer hip.loop's body func contains an inner hip.loop. Outer v_init
// is statically refined (the function arg is `tensor<256xf32>`) but
// outer body func arg slot 3 is declared `tensor<?xf32>`; inner
// hip.loop's v_init feeds from that block arg, so the inner's
// signature is also under-refined transitively. Phase 2 fixed-point
// iteration handles both levels: outer pass refines outer body arg ?xf32
// -> 256xf32, then inner pass (in the next iteration) sees its v_init
// type tighten to 256xf32 and refines inner_body's signature to match.
//
// `OnnxLoopOutlinePass` rejects nested `onnx.Loop` upstream so this
// shape is unreachable from production graphs, but pinning the
// contract here keeps the helper safe against future relaxations
// of that rejection.
//
// CHECK-LABEL: func.func private @inner_body
// CHECK-SAME:    %[[V:[^ ,]+]]: tensor<256xf32>
// CHECK-SAME:    -> (i32, tensor<256xf32>)
// CHECK:         return %{{.*}}, %[[V]] : i32, tensor<256xf32>
func.func private @inner_body(%ctx: !hip.context, %iter: tensor<i64>,
                              %cond_in: tensor<i1>,
                              %v: tensor<256xf32>,
                              %frame: !hip.loop_frame) -> (i32, tensor<256xf32>) {
  %status = arith.constant 0 : i32
  return %status, %v : i32, tensor<256xf32>
}

// CHECK-LABEL: func.func private @outer_body_with_inner_loop
// CHECK-SAME:    %[[V:[^ ,]+]]: tensor<256xf32>
// CHECK-SAME:    -> (i32, tensor<256xf32>)
// CHECK:         %[[R:.*]] = hip.loop
// CHECK-SAME:      iter_args(%[[V]] : tensor<256xf32>)
// CHECK-SAME:      -> (tensor<256xf32>)
// CHECK-SAME:      body @inner_body
// CHECK:         return %{{.*}}, %[[R]] : i32, tensor<256xf32>
func.func private @outer_body_with_inner_loop(%ctx: !hip.context,
                                              %iter: tensor<i64>,
                                              %cond_in: tensor<i1>,
                                              %v: tensor<256xf32>,
                                              %frame: !hip.loop_frame)
    -> (i32, tensor<256xf32>) {
  %status = arith.constant 0 : i32
  %M_inner = arith.constant 1 : index
  %cond_init_inner = arith.constant true
  %r = hip.loop(%ctx, %M_inner, %cond_init_inner)
                 iter_args(%v : tensor<256xf32>)
                 parent(%frame)
                 -> (tensor<256xf32>)
                 body @inner_body
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %status, %r : i32, tensor<256xf32>
}

// CHECK-LABEL: func.func @nested_loop_signatures
// CHECK:         %[[R:.*]] = hip.loop
// CHECK-SAME:      iter_args(%{{.*}} : tensor<256xf32>)
// CHECK-SAME:      -> (tensor<256xf32>)
// CHECK-SAME:      body @outer_body_with_inner_loop
// CHECK:         return %[[R]] : tensor<256xf32>
func.func @nested_loop_signatures(%ctx: !hip.context,
                                  %M: index, %cond: i1,
                                  %v_static: tensor<256xf32>) -> tensor<256xf32> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%v_static : tensor<256xf32>)
                 -> (tensor<256xf32>)
                 body @outer_body_with_inner_loop
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<256xf32>
}

// -----

// OneHot with a rank-0 (scalar) `depth` whose RUNTIME VALUE sizes the
// one-hot axis. The axis extent is data-dependent (here materialized by
// `hip.readback_scalar` into the DPS `outs` init, exactly as the
// ONNX->HIP converter emits it), so `--hip-infer-shapes` MUST leave the
// axis dim dynamic. Regression: `OneHotOp::reifyResultShapes` used to
// return a constant `1` for a rank-0 depth, so this pass narrowed the
// axis to a static `1` -- the scatter then dropped every index >= 1 and
// the axis collapsed to a single row. The non-axis dims still narrow
// from the static `indices` extents (2, 8).
// CHECK-LABEL: func.func @onehot_scalar_depth_axis_stays_dynamic
// CHECK:         hip.one_hot
// CHECK-SAME:      outs({{.*}} : tensor<2x8x?xf32>) {axis = 2 : i64} : tensor<2x8x?xf32>
// CHECK-NOT:     tensor<2x8x1xf32>
func.func @onehot_scalar_depth_axis_stays_dynamic(
    %ctx: !hip.context,
    %indices: tensor<2x8xi64>,
    %depth: tensor<i64>,
    %values: tensor<2xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %indices, %c0 : tensor<2x8xi64>
  %d1 = tensor.dim %indices, %c1 : tensor<2x8xi64>
  %d = hip.readback_scalar(%ctx, %depth : tensor<i64>) -> i64
  %di = arith.index_cast %d : i64 to index
  %init = tensor.empty(%d0, %d1, %di) : tensor<?x?x?xf32>
  %r = hip.one_hot(%ctx)
      ins(%indices, %depth, %values : tensor<2x8xi64>, tensor<i64>, tensor<2xf32>)
      outs(%init : tensor<?x?x?xf32>) {axis = 2 : i64} : tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}
