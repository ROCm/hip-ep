// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @div_f32(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %result = "onnx.Div"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @div_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.div(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>)

  func.func @div_i32(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi32> {
    %result = "onnx.Div"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %result : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @div_i32
  // CHECK: hip.div({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>) outs({{.*}} : tensor<4xi32>)

  func.func @div_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %result = "onnx.Div"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @div_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<?x?xf32>, %[[B3:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty({{.*}}) : tensor<?x?xf32>
  // CHECK: hip.div(%[[CTX3]]) ins(%[[A3]], %[[B3]] : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xf32>)

  func.func @div_broadcast_i64(%a: tensor<1xi64>, %b: tensor<32xi64>) -> tensor<32xi64> {
    %result = "onnx.Div"(%a, %b) : (tensor<1xi64>, tensor<32xi64>) -> tensor<32xi64>
    return %result : tensor<32xi64>
  }

  // CHECK-LABEL: func.func @div_broadcast_i64
  // CHECK: tensor.empty() : tensor<32xi64>
  // CHECK: hip.div(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1xi64>, tensor<32xi64>) outs({{.*}} : tensor<32xi64>)

  func.func @div_5d_broadcast(
      %a: tensor<2x3x4x5x6xi32>, %b: tensor<2x1x4x5x6xi32>)
      -> tensor<2x3x4x5x6xi32> {
    // CHECK-LABEL: func.func @div_5d_broadcast
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<2x3x4x5x6xi32> into tensor<2x3x4x30xi32>
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<2x1x4x5x6xi32> into tensor<2x1x4x30xi32>
    // CHECK: hip.div({{.*}}) ins({{.*}}, {{.*}} : tensor<2x3x4x30xi32>, tensor<2x1x4x30xi32>) outs({{.*}} : tensor<2x3x4x30xi32>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [2, 3, 4, 5, 6] : tensor<2x3x4x30xi32> into tensor<2x3x4x5x6xi32>
    %result = "onnx.Div"(%a, %b) :
        (tensor<2x3x4x5x6xi32>, tensor<2x1x4x5x6xi32>)
        -> tensor<2x3x4x5x6xi32>
    return %result : tensor<2x3x4x5x6xi32>
  }
}
