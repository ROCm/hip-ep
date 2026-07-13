// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %arg0 : tensor<4x8xf32>
  }

  func.func @quant_per_tensor(%x: tensor<4x8xf32>, %scale: tensor<f32>, %zp: tensor<ui8>) -> tensor<4x8xui8> {
    %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<4x8xf32>, tensor<f32>, tensor<ui8>) -> tensor<4x8xui8>
    return %y : tensor<4x8xui8>
  }

  // CHECK-LABEL: func.func @quant_per_tensor
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
  // CHECK: hip.quantize_linear(%[[CTX]]) ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<f32>)
  // CHECK: zero_points(%{{.*}} : tensor<ui8>)
  // CHECK: outs(%{{.*}} : tensor<4x8xui8>)
  // CHECK-NOT: onnx.QuantizeLinear

  func.func @quant_per_axis(%x: tensor<1x3x2x2xf32>, %scale: tensor<3xf32>, %zp: tensor<3xui8>) -> tensor<1x3x2x2xui8> {
    %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64} : (tensor<1x3x2x2xf32>, tensor<3xf32>, tensor<3xui8>) -> tensor<1x3x2x2xui8>
    return %y : tensor<1x3x2x2xui8>
  }

  // CHECK-LABEL: func.func @quant_per_axis
  // CHECK: hip.quantize_linear(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x3x2x2xf32>, tensor<3xf32>)
  // CHECK-NOT: onnx.QuantizeLinear

  func.func @quant_dynamic(%x: tensor<?x?xf32>, %scale: tensor<?xf32>) -> tensor<?x?xui8> {
    %y = "onnx.QuantizeLinear"(%x, %scale) {axis = 1 : si64} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xui8>
    return %y : tensor<?x?xui8>
  }

  // CHECK-LABEL: func.func @quant_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?xf32>, %[[SCALE:.*]]: tensor<?xf32>)
  // CHECK: %[[D0:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[D1:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xui8>
  // CHECK: hip.quantize_linear(%[[CTX]]) ins(%[[X]], %[[SCALE]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[INIT]] : tensor<?x?xui8>)
}
