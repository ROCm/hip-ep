// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s
// RUN: hip-mlir-opt --verify-each=0 --test-hip-dps-default-reify %s 2>&1 | FileCheck %s --check-prefix=MEMREF

// What this file tests
// --------------------
// `HipDpsOp::reifyResultShapes` -- the SHARED default body carried by the
// in-dialect `HipDpsOpInterface`. `Hip_DpsOp_AutoReify` emits a per-op
// `ReifyRankedShapedTypeOpInterface::reifyResultShapes` dispatcher that
// forwards to that default. The default walks
// `getDpsInits()` in tensor mode and lifts each init operand's runtime shape
// via `tensor::getMixedSizes`; memref mode has no SSA results and returns an
// empty list. This file pins that, for any tensor-mode operation selecting the
// default-reify family, reify produces:
//   - `IntegerAttr` for static dims (fold to `arith.constant`)
//   - `tensor.dim %outs, %i` for dynamic dims
//
// `hip.silu` is the worked example -- a single-init same-shape elementwise
// op with no per-op reify override on the #260 base. Other operations in the
// default-reify family follow the same contract.
//
// What this file does NOT test
// ----------------------------
// `MatmulOp::reifyResultShapes` -- matmul selects a manual-reify family with a
// tighter contract that lifts dims from the `A` / `B` ins operands. That is
// covered by `hip-matmul-reify-shapes.mlir`.

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

// -----

// Memref-mode DPS ops have zero SSA results. Exercise the shared HipDpsOp
// default body directly through the tool-only contract probe: it must return
// success with an empty reified list rather than one vector for the memref init.
// MEMREF-LABEL: func.func @default_reify_memref_mode
// MEMREF:         hip.cos
// MEMREF-SAME:      outs(%{{.*}} : memref<2x8xf32>)
// MEMREF-NOT:     tensor.dim
func.func @default_reify_memref_mode(%ctx: !hip.context,
                                     %x: memref<2x8xf32>,
                                     %y: memref<2x8xf32>) {
  hip.cos(%ctx) ins(%x : memref<2x8xf32>)
                outs(%y : memref<2x8xf32>)
                {"test.default_reify"}
  return
}

// -----

// Rank-zero success is one reified result with an empty dimension vector, not
// failure and not an empty result list.
// MEMREF: hip.where
// MEMREF-SAME: test.rank_zero_reified
func.func @rank_zero_reify(%ctx: !hip.context, %cond: tensor<i1>,
                           %x: tensor<f32>, %y: tensor<f32>,
                           %out: tensor<f32>) -> tensor<f32> {
  %result = hip.where(%ctx)
      ins(%cond, %x, %y : tensor<i1>, tensor<f32>, tensor<f32>)
      outs(%out : tensor<f32>)
      {test.reify_rank_zero} : tensor<f32>
  return %result : tensor<f32>
}

// -----

// Rank-zero semantic helpers must distinguish a successful empty shape from
// failure for Transpose, Gather, and GatherND.
// MEMREF-LABEL: func.func @transpose_rank_zero_reify
// MEMREF: hip.transpose
// MEMREF-SAME: test.rank_zero_reified
func.func @transpose_rank_zero_reify(
    %ctx: !hip.context, %input: tensor<f32>,
    %out: tensor<f32>) -> tensor<f32> {
  %result = hip.transpose(%ctx)
      ins(%input : tensor<f32>)
      outs(%out : tensor<f32>)
      {perm = [], test.reify_rank_zero}
      : tensor<f32>
  return %result : tensor<f32>
}

// MEMREF-LABEL: func.func @gather_rank_zero_reify
// MEMREF: hip.gather
// MEMREF-SAME: test.rank_zero_reified
func.func @gather_rank_zero_reify(
    %ctx: !hip.context, %data: tensor<4xf32>, %indices: tensor<i64>,
    %out: tensor<f32>) -> tensor<f32> {
  %result = hip.gather(%ctx)
      ins(%data, %indices : tensor<4xf32>, tensor<i64>)
      outs(%out : tensor<f32>)
      {axis = 0 : i64, test.reify_rank_zero}
      : tensor<f32>
  return %result : tensor<f32>
}

// MEMREF-LABEL: func.func @gather_nd_rank_zero_reify
// MEMREF: hip.gather_nd
// MEMREF-SAME: test.rank_zero_reified
func.func @gather_nd_rank_zero_reify(
    %ctx: !hip.context, %data: tensor<4xf32>,
    %indices: tensor<1xi64>, %out: tensor<f32>) -> tensor<f32> {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<4xf32>, tensor<1xi64>)
      outs(%out : tensor<f32>)
      {batch_dims = 0 : i64, test.reify_rank_zero}
      : tensor<f32>
  return %result : tensor<f32>
}

// -----

// A rank-mismatched later init must be diagnosed before the dynamic first
// output emits tensor.dim. The test pass temporarily adds one rank to init #1,
// invokes the shared default, and restores the valid type afterward.
// MEMREF: hip.layer_norm
// MEMREF-SAME: test.default_reify_failure_atomic_passed
func.func @default_reify_second_result_failure(
    %ctx: !hip.context, %input: tensor<?x?xf16>, %scale: tensor<?xf16>,
    %out: tensor<?x?xf16>, %mean: tensor<?x?xf32>,
    %inv_std: tensor<?x?xf32>)
    -> (tensor<?x?xf16>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %result:3 = hip.layer_norm(%ctx)
      ins(%input, %scale : tensor<?x?xf16>, tensor<?xf16>)
      outs(%out, %mean, %inv_std :
           tensor<?x?xf16>, tensor<?x?xf32>, tensor<?x?xf32>)
      {axis = -1 : i64, epsilon = 9.99999974e-06 : f32,
       stash_type = 1 : i64, test.default_reify_failure_atomic}
      : tensor<?x?xf16>, tensor<?x?xf32>, tensor<?x?xf32>
  return %result#0, %result#1, %result#2 :
      tensor<?x?xf16>, tensor<?x?xf32>, tensor<?x?xf32>
}
