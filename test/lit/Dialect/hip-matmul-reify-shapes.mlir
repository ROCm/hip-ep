// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for `Hip_MatmulOp::reifyResultShapes`.
//
// `reifyResultShapes` lifts the statically-known dims of `hip.matmul` into
// `OpFoldResult` (IntegerAttr for static dims; tensor.dim of the relevant
// operand for kDynamic dims). It is exercised here through the upstream
// `--resolve-shaped-type-result-dims` pass, which folds `tensor.dim` of an
// op result into either a constant (for static result dims) or a
// `tensor.dim` of an input (for dynamic dims that are resolved through
// reify to one of the input operands).
//
// The reify implementation honours the matmul shape contract:
//   M (penultimate dim)  -> dim of A at A.rank-2
//   N (last dim)         -> dim of B at B.rank-1
//   batch dim i          -> dim of A or B at the corresponding right-aligned
//                           position; if either side has size 1 (broadcast)
//                           or the side is out-of-range, the other side is
//                           used.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// --- 2D matmul, fully static result -> tensor.dim folds to constants. ---
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

// --- Dynamic batch (M-side comes from A); reify routes tensor.dim through A. ---
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

// --- Dynamic M (penultimate dim of A); reify routes tensor.dim of dim-0 to A. ---
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

// --- Dynamic N (last dim of B); reify routes tensor.dim of dim-1 to B. ---
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
