// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s
// RUN: hip-mlir-opt --test-hip-dps-default-reify %s | FileCheck %s --check-prefix=MEMREF

// What this file tests
// --------------------
// `Hip_DpsOp_SameShape` -- the parameterized TableGen base for ops whose
// result shape equals one named semantic source. It emits a per-op
// `ReifyRankedShapedTypeOpInterface::reifyResultShapes` dispatcher through
// `reifyElementwiseSameShape`, independent of the DPS init operand. This file
// pins that the generated dispatcher produces:
//   - `IntegerAttr` for static dims (fold to `arith.constant`)
//   - `tensor.dim %source, %i` for dynamic dims
//
// `hip.silu` is the input/output accessor-family example. Other migrated
// same-shape ops use the same generated mechanism with their own accessor
// stems.
//
// What this file does NOT test
// ----------------------------
// The shared outs-lift default and semantic shape contracts such as matmul are
// covered by their respective focused tests.

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

// Dynamic source operand: dim folds to `tensor.dim %x, %i`. Static dim folds
// to a constant.
// CHECK-LABEL: func.func @silu_dynamic
// CHECK-SAME:   (%{{[^,]*}}, %[[X:[A-Za-z0-9_]+]]: tensor<?x8xf16>,
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[D0:.*]] = tensor.dim %[[X]], %[[C0]] : tensor<?x8xf16>
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

// Mixed static + dynamic source: confirms per-dim handling (constant for
// static, tensor.dim for dynamic) on the same op invocation.
// CHECK-LABEL: func.func @silu_mixed
// CHECK-SAME:   (%{{[^,]*}}, %[[X:[A-Za-z0-9_]+]]: tensor<2x?x8xf16>,
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[D1:.*]] = tensor.dim %[[X]], %[[C1]] : tensor<2x?x8xf16>
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
