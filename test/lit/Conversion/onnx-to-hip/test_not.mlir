// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    return %arg0 : tensor<4xi1>
  }

  // --- Case 1: static shape ---
  func.func @not_static(%input: tensor<3x4xi1>) -> tensor<3x4xi1> {
    %result = "onnx.Not"(%input) : (tensor<3x4xi1>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @not_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xi1>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.not(%[[CTX]]) ins(%[[IN]] : tensor<3x4xi1>) outs({{.*}} : tensor<3x4xi1>)

  // --- Case 2: dynamic shape ---
  func.func @not_dynamic(%input: tensor<?x?xi1>) -> tensor<?x?xi1> {
    %result = "onnx.Not"(%input) : (tensor<?x?xi1>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @not_dynamic
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xi1>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi1>
  // CHECK: hip.not(%[[CTX2]]) ins(%[[ARG]] : tensor<?x?xi1>) outs(%[[INIT]] : tensor<?x?xi1>)
}
