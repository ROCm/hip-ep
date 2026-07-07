// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Greater(A, B) decomposes into Less(B, A). The original onnx.Greater must
  // disappear.
  func.func @greater_static(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %r : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @greater_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.less(%[[CTX]]) ins(%[[B]], %[[A]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)
  // CHECK-NOT: onnx.Greater

  func.func @greater_i32(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<4xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
    return %r : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @greater_i32
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi32>, tensor<4xi32>) outs({{.*}} : tensor<4xi1>)
  // CHECK-NOT: onnx.Greater

  func.func @greater_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %r = "onnx.Greater"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %r : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @greater_dynamic
  // CHECK: tensor.dim
  // CHECK: hip.less({{.*}}) ins({{.*}}, {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)
  // CHECK-NOT: onnx.Greater
}
