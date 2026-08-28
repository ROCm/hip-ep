// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @less_f32(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %r = "onnx.Less"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %r : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @less_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.less(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)

  func.func @less_i32(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
    %r = "onnx.Less"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    return %r : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @less_i32
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>) outs({{.*}} : tensor<4xi1>)

  func.func @less_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %r = "onnx.Less"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %r : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @less_dynamic
  // CHECK: tensor.empty
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)
}
