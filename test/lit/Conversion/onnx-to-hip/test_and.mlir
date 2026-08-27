// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xi1>) -> tensor<4xi1> {
    return %arg0 : tensor<4xi1>
  }

  func.func @and_static(%a: tensor<3x4xi1>, %b: tensor<3x4xi1>) -> tensor<3x4xi1> {
    %result = "onnx.And"(%a, %b) : (tensor<3x4xi1>, tensor<3x4xi1>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @and_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xi1>, %[[B:.*]]: tensor<3x4xi1>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.and(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xi1>, tensor<3x4xi1>) outs({{.*}} : tensor<3x4xi1>)

  func.func @and_broadcast(%a: tensor<3x4xi1>, %b: tensor<4xi1>) -> tensor<3x4xi1> {
    %result = "onnx.And"(%a, %b) : (tensor<3x4xi1>, tensor<4xi1>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @and_broadcast
  // CHECK: hip.and({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xi1>, tensor<4xi1>) outs({{.*}} : tensor<3x4xi1>)

  func.func @and_dynamic(%a: tensor<?x?xi1>, %b: tensor<?x?xi1>) -> tensor<?x?xi1> {
    %result = "onnx.And"(%a, %b) : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @and_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<?x?xi1>, %[[B3:.*]]: tensor<?x?xi1>)
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty({{.*}}) : tensor<?x?xi1>
  // CHECK: hip.and(%[[CTX3]]) ins(%[[A3]], %[[B3]] : tensor<?x?xi1>, tensor<?x?xi1>) outs({{.*}} : tensor<?x?xi1>)

  func.func @and_5d(
      %a: tensor<4x1x6x8x1xi1>, %b: tensor<4x1x6x8x1xi1>)
      -> tensor<4x1x6x8x1xi1> {
    // CHECK-LABEL: func.func @and_5d
    // CHECK-SAME: (%[[CTX4:.*]]: !hip.context, %[[A4:.*]]: tensor<4x1x6x8x1xi1>, %[[B4:.*]]: tensor<4x1x6x8x1xi1>)
    // CHECK-NOT: onnx.And
    // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<4x1x6x8x1xi1> into tensor<4x1x6x8xi1>
    // CHECK: hip.and(%[[CTX4]]) ins({{.*}}, {{.*}} : tensor<4x1x6x8xi1>, tensor<4x1x6x8xi1>) outs({{.*}} : tensor<4x1x6x8xi1>)
    // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 1, 6, 8, 1] : tensor<4x1x6x8xi1> into tensor<4x1x6x8x1xi1>
    %r = "onnx.And"(%a, %b) : (tensor<4x1x6x8x1xi1>, tensor<4x1x6x8x1xi1>) -> tensor<4x1x6x8x1xi1>
    return %r : tensor<4x1x6x8x1xi1>
  }
}
