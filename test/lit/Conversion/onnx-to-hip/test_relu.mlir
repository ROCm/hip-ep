// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @relu_static_f32(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %y = "onnx.Relu"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @relu_static_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>)
  // CHECK-NOT: onnx.Relu
  // The zero literal is synthesized during compute conversion, so this
  // carrier proves the second constant sweep ran.
  // CHECK: hip.constant {serialization_order = 0 : i64, value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: hip.max(%[[CTX]]) ins(%[[X]]

  func.func @relu_static_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
    %y = "onnx.Relu"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
    return %y : tensor<2x3xf16>
  }

  // CHECK-LABEL: func.func @relu_static_f16
  // CHECK-NOT: onnx.Relu
  // CHECK: hip.max

  func.func @relu_dynamic(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %y = "onnx.Relu"(%x) : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @relu_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<?x?xf32>)
  // CHECK-NOT: onnx.Relu
  // CHECK: hip.max(%[[CTX3]])
}
