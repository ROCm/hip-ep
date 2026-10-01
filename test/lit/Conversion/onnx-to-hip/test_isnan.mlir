// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Frontend bool encoding used by the Distil decoder: f16 -> ui8.
  func.func @isnan_f16_ui8(%input: tensor<24x20x1x1xf16>) -> tensor<24x20x1x1xui8> {
    %result = "onnx.IsNaN"(%input) : (tensor<24x20x1x1xf16>) -> tensor<24x20x1x1xui8>
    return %result : tensor<24x20x1x1xui8>
  }

  // CHECK-LABEL: func.func @isnan_f16_ui8
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<24x20x1x1xf16>)
  // CHECK: tensor.empty() : tensor<24x20x1x1xui8>
  // CHECK: hip.isnan(%[[CTX]]) ins(%[[IN]] : tensor<24x20x1x1xf16>) outs({{.*}} : tensor<24x20x1x1xui8>)

  // Hand-written bool encoding: f32 -> i1.
  func.func @isnan_f32_i1(%input: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %result = "onnx.IsNaN"(%input) : (tensor<3x4xf32>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @isnan_f32_i1
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[IN2:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.isnan(%[[CTX2]]) ins(%[[IN2]] : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)

  func.func @isnan_dynamic(%input: tensor<?x?xf16>) -> tensor<?x?xui8> {
    %result = "onnx.IsNaN"(%input) : (tensor<?x?xf16>) -> tensor<?x?xui8>
    return %result : tensor<?x?xui8>
  }

  // CHECK-LABEL: func.func @isnan_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xui8>
  // CHECK: hip.isnan(%[[CTX3]]) ins(%[[ARG]] : tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xui8>)
}
