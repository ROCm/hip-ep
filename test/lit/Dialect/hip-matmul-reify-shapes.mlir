// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

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

// Pin the "dynamic + static>1 -> static" tightening: A's batch dim is dynamic,
// B's is static-5; per NumPy / TF / ONNX broadcast contract A must be 1 or 5
// at runtime. Reify therefore picks the static side, so tensor.dim of the
// result's batch dim folds to constant 5 (CHECK-NOT: tensor.dim) — even though
// the result type is left as ?x4x16. Delegated to upstream
// mlir::OpTrait::util::getBroadcastedShape.
// CHECK-LABEL: func.func @reify_dyn_static_batch_broadcast
// CHECK-NOT:   tensor.dim
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       return %[[C5]], %[[C4]], %[[C16]]
func.func @reify_dyn_static_batch_broadcast(%ctx: !hip.context,
                                            %a: tensor<?x4x8xf16>,
                                            %b: tensor<5x8x16xf16>,
                                            %c: tensor<?x4x16xf16>) -> (index, index, index) {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<?x4x8xf16>, tensor<5x8x16xf16>)
    outs(%c : tensor<?x4x16xf16>) : tensor<?x4x16xf16>
  %d0_idx = arith.constant 0 : index
  %d1_idx = arith.constant 1 : index
  %d2_idx = arith.constant 2 : index
  %d0 = tensor.dim %r, %d0_idx : tensor<?x4x16xf16>
  %d1 = tensor.dim %r, %d1_idx : tensor<?x4x16xf16>
  %d2 = tensor.dim %r, %d2_idx : tensor<?x4x16xf16>
  return %d0, %d1, %d2 : index, index, index
}
