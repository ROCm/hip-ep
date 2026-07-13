// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4x8xi8>) -> tensor<4x8xi8> {
    return %arg0 : tensor<4x8xi8>
  }

  func.func @dequant_per_tensor(%x: tensor<4x8xi8>, %scale: tensor<f32>, %zp: tensor<i8>) -> tensor<4x8xf32> {
    %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<4x8xi8>, tensor<f32>, tensor<i8>) -> tensor<4x8xf32>
    return %y : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @dequant_per_tensor
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
  // CHECK: hip.dequantize_linear(%[[CTX]]) ins(%{{.*}}, %{{.*}} : tensor<4x8xi8>, tensor<f32>)
  // CHECK: zero_points(%{{.*}} : tensor<i8>)
  // CHECK: outs(%{{.*}} : tensor<4x8xf32>)
  // CHECK-NOT: onnx.DequantizeLinear

  func.func @dequant_per_axis(%x: tensor<1x3x2x2xui8>, %scale: tensor<3xf32>, %zp: tensor<3xui8>) -> tensor<1x3x2x2xf32> {
    %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<1x3x2x2xui8>, tensor<3xf32>, tensor<3xui8>) -> tensor<1x3x2x2xf32>
    return %y : tensor<1x3x2x2xf32>
  }

  // CHECK-LABEL: func.func @dequant_per_axis
  // CHECK: hip.dequantize_linear(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x3x2x2xui8>, tensor<3xf32>)
  // CHECK-NOT: onnx.DequantizeLinear

  func.func @dequant_dynamic(%x: tensor<?x?xi8>, %scale: tensor<?xf32>) -> tensor<?x?xf32> {
    %y = "onnx.DequantizeLinear"(%x, %scale) {axis = 1 : si64} : (tensor<?x?xi8>, tensor<?xf32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @dequant_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?xi8>, %[[SCALE:.*]]: tensor<?xf32>)
  // CHECK: %[[D0:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xi8>
  // CHECK: %[[D1:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xi8>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
  // CHECK: hip.dequantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<?x?xi8>, tensor<?xf32>) outs(%[[INIT]] : tensor<?x?xf32>)
}
