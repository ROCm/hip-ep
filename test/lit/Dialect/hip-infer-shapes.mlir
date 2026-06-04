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
