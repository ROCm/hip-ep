// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// What this file tests
// --------------------
// `MatmulOp::reifyResultShapes` (in lib/Dialect/IR/HipReifyResultShapesImpl.cpp)
// — i.e. the PRODUCER side of `ReifyRankedShapedTypeOpInterface` for
// `hip.matmul`. Validates that for any operand-shape combination (static,
// dynamic batch, dynamic M, dynamic N, broadcast) it returns a correct
// per-dim `OpFoldResult`: an `IntegerAttr` for static dims, and a
// `tensor.dim` of the right operand at the right local index for dynamic dims.
//
// What this file does NOT test
// ----------------------------
// `--hip-infer-shapes` — the production pass that consumes these
// `OpFoldResult`s to refine result types and rebuild `tensor.empty`
// producers. Pass-level behavior (composeRefinedShape, cast barriers,
// DPS-init exemption) is covered by `hip-infer-shapes.mlir`.
//
// Why this test exists in addition to the pass test
// -------------------------------------------------
// `--hip-infer-shapes` only consumes the *constant* branch of each
// `OpFoldResult` (via `getConstantIntValue`); it silently discards
// dynamic ones. A bug in reify's dynamic-dim source-picking (wrong
// operand chosen, wrong local dim index, or a malformed `tensor.dim`
// emitted) is therefore invisible to the pass test. Upstream's
// `--resolve-shaped-type-result-dims` materializes every `OpFoldResult`
// — static and dynamic — into IR that FileCheck can inspect, exposing
// those bugs. The upstream pass is used here purely as a generic
// producer-contract validator; it is not part of the production
// pipeline.

// CHECK-LABEL: func.func @reify_2d_static
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       return %[[C2]], %[[C8]]
func.func @reify_2d_static(%ctx: !hip.context,
                           %a: tensor<2x4xf16>,
                           %b: tensor<4x8xf16>,
                           %c: tensor<2x8xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<2x8xf16>) : tensor<2x8xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<2x8xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<2x8xf16>
  return %d0, %d1 : index, index
}

// -----

// Dynamic contraction extents are runtime-checked and do not contribute to the
// result shape. Reification therefore emits only M/N dimensions.
// CHECK-LABEL: func.func @reify_dynamic_k_both
// CHECK-SAME: %[[A:[A-Za-z0-9_]+]]: tensor<?x?xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<?x?xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK: %[[N:.*]] = tensor.dim %[[B]], %[[C1]]
// CHECK: return %[[M]], %[[N]]
func.func @reify_dynamic_k_both(
    %ctx: !hip.context, %a: tensor<?x?xf16>, %b: tensor<?x?xf16>,
    %out: tensor<?x?xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
    outs(%out : tensor<?x?xf16>) : tensor<?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_k_a_only
// CHECK: return
func.func @reify_dynamic_k_a_only(
    %ctx: !hip.context, %a: tensor<2x?xf16>, %b: tensor<4x8xf16>,
    %out: tensor<2x8xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x?xf16>, tensor<4x8xf16>)
    outs(%out : tensor<2x8xf16>) : tensor<2x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<2x8xf16>
  %d1 = tensor.dim %r, %c1 : tensor<2x8xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_k_b_only
// CHECK: return
func.func @reify_dynamic_k_b_only(
    %ctx: !hip.context, %a: tensor<2x4xf16>, %b: tensor<?x8xf16>,
    %out: tensor<2x8xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<?x8xf16>)
    outs(%out : tensor<2x8xf16>) : tensor<2x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<2x8xf16>
  %d1 = tensor.dim %r, %c1 : tensor<2x8xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_batch
// CHECK-SAME:   %[[A:[A-Za-z0-9_]+]]: tensor<?x4x8xf16>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[B0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x4x8xf16>
// CHECK:       return %[[B0]], %[[C4]], %[[C16]]
func.func @reify_dynamic_batch(%ctx: !hip.context,
                               %a: tensor<?x4x8xf16>,
                               %b: tensor<8x16xf16>,
                               %c: tensor<?x4x16xf16>) -> (index, index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x4x8xf16>, tensor<8x16xf16>)
    outs(%c : tensor<?x4x16xf16>) : tensor<?x4x16xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d2_idx = arith.constant 2 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<?x4x16xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<?x4x16xf16>
  %d2 = tensor.dim %r, %d2_idx : tensor<?x4x16xf16>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_M
// CHECK-SAME:   %[[A:[A-Za-z0-9_]+]]: tensor<?x4xf16>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[M:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x4xf16>
// CHECK:       return %[[M]], %[[C8]]
func.func @reify_dynamic_M(%ctx: !hip.context,
                           %a: tensor<?x4xf16>,
                           %b: tensor<4x8xf16>,
                           %c: tensor<?x8xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<?x8xf16>) : tensor<?x8xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<?x8xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<?x8xf16>
  return %d0, %d1 : index, index
}

// -----

// CHECK-LABEL: func.func @reify_dynamic_N
// CHECK-SAME:   %[[A:[A-Za-z0-9_]+]]: tensor<2x4xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<4x?xf16>
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[N:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<4x?xf16>
// CHECK:       return %[[C2]], %[[N]]
func.func @reify_dynamic_N(%ctx: !hip.context,
                           %a: tensor<2x4xf16>,
                           %b: tensor<4x?xf16>,
                           %c: tensor<2x?xf16>) -> (index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x?xf16>)
    outs(%c : tensor<2x?xf16>) : tensor<2x?xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<2x?xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<2x?xf16>
  return %d0, %d1 : index, index
}

// -----

// Gemma-shaped multi-axis dynamic batch. Each broadcasted batch extent is
// reified independently; runtime matrix-count validation handles layouts that
// turn out to be partial broadcasts.
// CHECK-LABEL: func.func @reify_gemma_dynamic_batches
// CHECK: %[[BATCH0:.*]] = arith.select
// CHECK: %[[BATCH1:.*]] = arith.select
// CHECK: return %[[BATCH0]], %[[BATCH1]], %{{.*}}, %{{.*}}
func.func @reify_gemma_dynamic_batches(
    %ctx: !hip.context,
    %a: tensor<?x?x4x8xf16>,
    %b: tensor<?x?x8x16xf16>,
    %c: tensor<?x?x4x16xf16>) -> (index, index, index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x?x4x8xf16>, tensor<?x?x8x16xf16>)
    outs(%c : tensor<?x?x4x16xf16>) : tensor<?x?x4x16xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?x4x16xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?x4x16xf16>
  %d2 = tensor.dim %r, %c2 : tensor<?x?x4x16xf16>
  %d3 = tensor.dim %r, %c3 : tensor<?x?x4x16xf16>
  return %d0, %d1, %d2, %d3 : index, index, index, index
}

// -----

// Whole-matrix broadcast remains safe with a dynamic batch: A contains one
// matrix and B supplies every output batch, so no runtime-only validation is
// needed.
// CHECK-LABEL: func.func @reify_dynamic_whole_matrix_broadcast
// CHECK-SAME: %[[B:[A-Za-z0-9_]+]]: tensor<?x8x16xf16>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[BATCH:.*]] = tensor.dim %[[B]], %[[C0]]
// CHECK:       return %[[BATCH]], %[[C4]], %[[C16]]
func.func @reify_dynamic_whole_matrix_broadcast(%ctx: !hip.context,
                                            %a: tensor<1x4x8xf16>,
                                            %b: tensor<?x8x16xf16>,
                                            %c: tensor<?x4x16xf16>) -> (index, index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<1x4x8xf16>, tensor<?x8x16xf16>)
    outs(%c : tensor<?x4x16xf16>) : tensor<?x4x16xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d2_idx = arith.constant 2 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<?x4x16xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<?x4x16xf16>
  %d2 = tensor.dim %r, %d2_idx : tensor<?x4x16xf16>
  return %d0, %d1, %d2 : index, index, index
}

// -----

// Unequal ranks: B supplies the leading batch dim, A supplies M, B supplies N.
// CHECK-LABEL: func.func @reify_2d_3d
// CHECK-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?x4xf16>, %[[B:[A-Za-z0-9_]+]]: tensor<?x4x?xf16>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[BATCH:.*]] = tensor.dim %[[B]], %[[C0]]
// CHECK-DAG: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[B]], %[[C2]]
// CHECK: return %[[BATCH]], %[[M]], %[[N]]
func.func @reify_2d_3d(%ctx: !hip.context,
                       %a: tensor<?x4xf16>,
                       %b: tensor<?x4x?xf16>,
                       %c: tensor<?x?x?xf16>) -> (index, index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x4xf16>, tensor<?x4x?xf16>)
    outs(%c : tensor<?x?x?xf16>) : tensor<?x?x?xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?x?xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x?x?xf16>
  %d2 = tensor.dim %r, %c2 : tensor<?x?x?xf16>
  return %d0, %d1, %d2 : index, index, index
}
