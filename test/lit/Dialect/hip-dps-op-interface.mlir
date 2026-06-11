// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// What this file tests
// --------------------
// `HipDpsOp::reifyResultShapes` -- the SHARED default body carried by the
// in-dialect `HipDpsOpInterface`. The TableGen base class `Hip_DpsOp`
// auto-emits a per-op `ReifyRankedShapedTypeOpInterface::reifyResultShapes`
// dispatcher that forwards to that default. The default walks
// `getDpsInits()` and lifts each init operand's runtime shape via
// `tensor::getMixedSizes` (tensor mode) / `memref::getMixedSizes` (memref
// mode). This file pins that, for any non-opt-out op (i.e. autoReify=1, the
// default), reify produces:
//   - `IntegerAttr` for static dims (fold to `arith.constant`)
//   - `tensor.dim %outs, %i` for dynamic dims
//
// `hip.silu` is the worked example -- a single-init same-shape elementwise
// op with no per-op reify override on the #260 base. Other Hip_DpsOps with
// the default (silu, sigmoid, rope, rms_norm, ...) follow the same contract
// because they share the same auto-emitted dispatcher.
//
// What this file does NOT test
// ----------------------------
// `MatmulOp::reifyResultShapes` -- matmul opts out of the default
// (`autoReify=0`) and provides a tighter contract that lifts dims from the
// `A` / `B` ins operands instead of the outs. That is covered by
// `hip-matmul-reify-shapes.mlir`.

// CHECK-LABEL: func.func @silu_static
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       return %[[C2]], %[[C8]]
func.func @silu_static(%ctx: !hip.context,
                       %x: tensor<2x8xf16>,
                       %y: tensor<2x8xf16>) -> (index, index) {
  %r = hip.silu(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%y : tensor<2x8xf16>) -> tensor<2x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<2x8xf16>
  %d1 = tensor.dim %r, %c1 : tensor<2x8xf16>
  return %d0, %d1 : index, index
}

// -----

// Dynamic outs operand: dim folds to `tensor.dim %y, %i`. Static dim folds
// to a constant. The `, %[[Y...]]: tensor<?x8xf16>)` regex anchors to the
// LAST arg of `tensor<?x8xf16>` shape (i.e. the outs operand), since the
// input shares the same shape and FileCheck would otherwise capture the
// first one.
// CHECK-LABEL: func.func @silu_dynamic
// CHECK-SAME:   , %[[Y:[A-Za-z0-9_]+]]: tensor<?x8xf16>)
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[D0:.*]] = tensor.dim %[[Y]], %[[C0]] : tensor<?x8xf16>
// CHECK:       return %[[D0]], %[[C8]]
func.func @silu_dynamic(%ctx: !hip.context,
                        %x: tensor<?x8xf16>,
                        %y: tensor<?x8xf16>) -> (index, index) {
  %r = hip.silu(%ctx) ins(%x : tensor<?x8xf16>)
                      outs(%y : tensor<?x8xf16>) -> tensor<?x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x8xf16>
  %d1 = tensor.dim %r, %c1 : tensor<?x8xf16>
  return %d0, %d1 : index, index
}

// -----

// Mixed static + dynamic outs: confirms per-dim handling (constant for
// static, tensor.dim for dynamic) on the same op invocation.
// CHECK-LABEL: func.func @silu_mixed
// CHECK-SAME:   , %[[Y:[A-Za-z0-9_]+]]: tensor<2x?x8xf16>)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[D1:.*]] = tensor.dim %[[Y]], %[[C1]] : tensor<2x?x8xf16>
// CHECK:       return %[[C2]], %[[D1]], %[[C8]]
func.func @silu_mixed(%ctx: !hip.context,
                      %x: tensor<2x?x8xf16>,
                      %y: tensor<2x?x8xf16>) -> (index, index, index) {
  %r = hip.silu(%ctx) ins(%x : tensor<2x?x8xf16>)
                      outs(%y : tensor<2x?x8xf16>) -> tensor<2x?x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %r, %c0 : tensor<2x?x8xf16>
  %d1 = tensor.dim %r, %c1 : tensor<2x?x8xf16>
  %d2 = tensor.dim %r, %c2 : tensor<2x?x8xf16>
  return %d0, %d1, %d2 : index, index, index
}
