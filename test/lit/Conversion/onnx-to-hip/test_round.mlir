// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @round_f32(%input: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %result = "onnx.Round"(%input) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @round_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.round(%[[CTX]]) ins(%[[IN]] : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>)

  func.func @round_dynamic(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = "onnx.Round"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @round_dynamic
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
  // CHECK: hip.round(%[[CTX2]]) ins(%[[ARG]] : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
}
