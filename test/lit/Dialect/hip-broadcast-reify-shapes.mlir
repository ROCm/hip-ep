// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s --check-prefix=REIFY
// RUN: hip-mlir-opt --hip-infer-shapes %s | FileCheck %s --check-prefix=INFER

// Dynamic/dynamic broadcast must select the non-unit runtime extent.
// REIFY-LABEL: func.func @dynamic_dynamic
// REIFY-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?xf32>, %[[B:[A-Za-z0-9_]+]]: tensor<?xf32>
// REIFY-DAG: %[[C0:.*]] = arith.constant 0 : index
// REIFY-DAG: %[[C1:.*]] = arith.constant 1 : index
// REIFY-DAG: %[[AD:.*]] = tensor.dim %[[A]], %[[C0]]
// REIFY-DAG: %[[BD:.*]] = tensor.dim %[[B]], %[[C0]]
// REIFY: %[[IS1:.*]] = arith.cmpi eq, %[[AD]], %[[C1]] : index
// REIFY: %[[D:.*]] = arith.select %[[IS1]], %[[BD]], %[[AD]] : index
// REIFY: return %[[D]]
func.func @dynamic_dynamic(%ctx: !hip.context, %a: tensor<?xf32>,
                           %b: tensor<?xf32>, %out: tensor<?xf32>) -> index {
  %r = hip.add(%ctx)
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) -> tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d = tensor.dim %r, %c0 : tensor<?xf32>
  return %d : index
}

// Max uses the same operand-derived broadcast contract as Min/Add rather than
// lifting a potentially stale extent from its destination.
// REIFY-LABEL: func.func @max_dynamic_dynamic
// REIFY-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?xf32>, %[[B:[A-Za-z0-9_]+]]: tensor<?xf32>
// REIFY-DAG: %[[C0:.*]] = arith.constant 0 : index
// REIFY-DAG: %[[C1:.*]] = arith.constant 1 : index
// REIFY-DAG: %[[AD:.*]] = tensor.dim %[[A]], %[[C0]]
// REIFY-DAG: %[[BD:.*]] = tensor.dim %[[B]], %[[C0]]
// REIFY: %[[IS1:.*]] = arith.cmpi eq, %[[AD]], %[[C1]] : index
// REIFY: %[[D:.*]] = arith.select %[[IS1]], %[[BD]], %[[AD]] : index
// REIFY: return %[[D]]
func.func @max_dynamic_dynamic(%ctx: !hip.context, %a: tensor<?xf32>,
                               %b: tensor<?xf32>,
                               %out: tensor<?xf32>) -> index {
  %r = hip.max(%ctx)
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d = tensor.dim %r, %c0 : tensor<?xf32>
  return %d : index
}

// Each asymmetric axis comes directly from the operand that spans it.
// REIFY-LABEL: func.func @asymmetric_axes
// REIFY-SAME: (%{{.*}}: !hip.context, %[[A:[A-Za-z0-9_]+]]: tensor<?x1xf32>, %[[B:[A-Za-z0-9_]+]]: tensor<1x?xf32>
// REIFY-DAG: %[[C0:.*]] = arith.constant 0 : index
// REIFY-DAG: %[[C1:.*]] = arith.constant 1 : index
// REIFY: %[[M:.*]] = tensor.dim %[[A]], %[[C0]]
// REIFY: %[[N:.*]] = tensor.dim %[[B]], %[[C1]]
// REIFY: return %[[M]], %[[N]]
func.func @asymmetric_axes(%ctx: !hip.context, %a: tensor<?x1xf32>,
                           %b: tensor<1x?xf32>,
                           %out: tensor<?x?xf32>) -> (index, index) {
  %r = hip.max(%ctx)
    ins(%a, %b : tensor<?x1xf32>, tensor<1x?xf32>)
    outs(%out : tensor<?x?xf32>) : tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %r, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %r, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}

// Broadcasting zero with one yields zero; max(0, 1) would be incorrect.
// REIFY-LABEL: func.func @zero_one
// REIFY-DAG: %[[C0:.*]] = arith.constant 0 : index
// REIFY: return %[[C0]]
func.func @zero_one(%ctx: !hip.context, %a: tensor<0xf32>,
                    %b: tensor<1xf32>, %out: tensor<?xf32>) -> index {
  %r = hip.min(%ctx)
    ins(%a, %b : tensor<0xf32>, tensor<1xf32>)
    outs(%out : tensor<?xf32>) : tensor<?xf32>
  %c0 = arith.constant 0 : index
  %d = tensor.dim %r, %c0 : tensor<?xf32>
  return %d : index
}

// Rank-zero broadcast is a successful empty reified shape. hip-infer-shapes
// must not fail merely because the result has no dimensions.
// INFER-LABEL: func.func @rank_zero
// INFER: %[[R:.*]] = hip.where
// INFER: return %[[R]] : tensor<f32>
func.func @rank_zero(%ctx: !hip.context, %cond: tensor<i1>,
                     %x: tensor<f32>, %y: tensor<f32>,
                     %out: tensor<f32>) -> tensor<f32> {
  %r = hip.where(%ctx)
    ins(%cond, %x, %y : tensor<i1>, tensor<f32>, tensor<f32>)
    outs(%out : tensor<f32>) : tensor<f32>
  return %r : tensor<f32>
}
