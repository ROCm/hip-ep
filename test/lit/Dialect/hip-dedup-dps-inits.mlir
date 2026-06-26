// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-dedup-dps-inits %s | FileCheck %s

// Global-value-numbering merge: two DPS-init empties whose dynamic extents are
// computed by structurally-identical but SSA-DISTINCT chains (`tensor.dim %a`
// at two textual sites) must be recognised as the same size and merged when
// their live ranges are disjoint. A chained sequence makes the ranges disjoint:
// %e0's buffer (aliased by %y0) dies once %y0 is read by the middle matmul, and
// %e1 is created only afterwards. The intermediate empty %er stays distinct
// (its range overlaps both). This is the cross-layer reuse a stock CSE achieves
// only by folding the shape arithmetic first; the dedup pass proves it via GVN
// without merging simultaneously-live buffers.
// CHECK-LABEL: func.func @gvn_merge_disjoint
func.func @gvn_merge_disjoint(%ctx: !hip.context, %a: tensor<?x4xf16>,
                              %b: tensor<4x8xf16>, %bb: tensor<8x8xf16>)
    -> tensor<?x8xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x4xf16>
  // CHECK: %[[E0:.*]] = tensor.empty(%{{.*}}) : tensor<?x8xf16>
  %e0 = tensor.empty(%d0) : tensor<?x8xf16>
  %y0 = hip.matmul(%ctx) ins(%a, %b : tensor<?x4xf16>, tensor<4x8xf16>)
    outs(%e0 : tensor<?x8xf16>) : tensor<?x8xf16>
  // Intermediate buffer (overlaps both ranges) — stays distinct.
  // CHECK: tensor.empty(%{{.*}}) : tensor<?x8xf16>
  %er = tensor.empty(%d0) : tensor<?x8xf16>
  %r = hip.matmul(%ctx) ins(%y0, %bb : tensor<?x8xf16>, tensor<8x8xf16>)
    outs(%er : tensor<?x8xf16>) : tensor<?x8xf16>
  // %d1 is a fresh SSA value structurally identical to %d0 -> GVN-equal.
  %d1 = tensor.dim %a, %c0 : tensor<?x4xf16>
  // No third empty here: %e1 was merged onto %[[E0]], reused as this matmul's
  // destination because [%e0..read-of-%y0] and [%e1..] are disjoint.
  // CHECK-NOT: tensor.empty
  %e1 = tensor.empty(%d1) : tensor<?x8xf16>
  // CHECK: hip.matmul(%{{.*}}) ins(%{{.*}}) outs(%[[E0]] : tensor<?x8xf16>)
  %y1 = hip.matmul(%ctx) ins(%r, %bb : tensor<?x8xf16>, tensor<8x8xf16>)
    outs(%e1 : tensor<?x8xf16>) : tensor<?x8xf16>
  return %y1 : tensor<?x8xf16>
}

// -----

// Negative: same static type, but extents derived from DIFFERENT dynamic
// tensors (%a vs %a2) are NOT provably equal, so the empties must stay distinct
// even though their live ranges are disjoint. Guards against an over-eager GVN
// that ignores the size operands.
// CHECK-LABEL: func.func @no_merge_different_size
func.func @no_merge_different_size(%ctx: !hip.context, %a: tensor<?x4xf16>,
                                   %a2: tensor<?x4xf16>, %b: tensor<4x8xf16>,
                                   %bb: tensor<8x8xf16>) -> tensor<?x8xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x4xf16>
  // CHECK: tensor.empty(%{{.*}}) : tensor<?x8xf16>
  %e0 = tensor.empty(%d0) : tensor<?x8xf16>
  %y0 = hip.matmul(%ctx) ins(%a, %b : tensor<?x4xf16>, tensor<4x8xf16>)
    outs(%e0 : tensor<?x8xf16>) : tensor<?x8xf16>
  %er = tensor.empty(%d0) : tensor<?x8xf16>
  %r = hip.matmul(%ctx) ins(%y0, %bb : tensor<?x8xf16>, tensor<8x8xf16>)
    outs(%er : tensor<?x8xf16>) : tensor<?x8xf16>
  // Different source tensor -> GVN-distinct extent -> keep separate.
  %d1 = tensor.dim %a2, %c0 : tensor<?x4xf16>
  // CHECK: tensor.empty(%{{.*}}) : tensor<?x8xf16>
  %e1 = tensor.empty(%d1) : tensor<?x8xf16>
  %y1 = hip.matmul(%ctx) ins(%r, %bb : tensor<?x8xf16>, tensor<8x8xf16>)
    outs(%e1 : tensor<?x8xf16>) : tensor<?x8xf16>
  return %y1 : tensor<?x8xf16>
}
