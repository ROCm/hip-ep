// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @isinf_f32(%input: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %result = "onnx.IsInf"(%input) {detect_negative = 1 : si64, detect_positive = 1 : si64}
        : (tensor<3x4xf32>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @isinf_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.isinf(%[[CTX]]) ins(%[[IN]] : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)

  func.func @isinf_f16(%input: tensor<128xf16>) -> tensor<128xi1> {
    %result = "onnx.IsInf"(%input) : (tensor<128xf16>) -> tensor<128xi1>
    return %result : tensor<128xi1>
  }

  // CHECK-LABEL: func.func @isinf_f16
  // CHECK: hip.isinf(%[[CTX2:.*]]) ins({{.*}} : tensor<128xf16>) outs({{.*}} : tensor<128xi1>)

  func.func @isinf_detect_pos_only(%input: tensor<4xf32>) -> tensor<4xi1> {
    %result = "onnx.IsInf"(%input) {detect_negative = 0 : si64, detect_positive = 1 : si64}
        : (tensor<4xf32>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @isinf_detect_pos_only
  // CHECK: hip.isinf(%{{.*}}) ins({{.*}} : tensor<4xf32>) outs({{.*}} : tensor<4xi1>) {detect_negative = 0 : i64}

  func.func @isinf_dynamic(%input: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %result = "onnx.IsInf"(%input) : (tensor<?x?xf32>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @isinf_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xi1>
  // CHECK: hip.isinf(%[[CTX3]]) ins(%[[ARG]] : tensor<?x?xf32>) outs(%[[INIT]] : tensor<?x?xi1>)
}
