// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @equal_f32(%a: tensor<3x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @equal_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<3x4xf32>, %[[B:.*]]: tensor<3x4xf32>)
  // CHECK: tensor.empty() : tensor<3x4xi1>
  // CHECK: hip.equal(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<3x4xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<3x4xi1>)

  func.func @equal_i64(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
    return %result : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @equal_i64
  // CHECK: hip.equal({{.*}}) ins({{.*}}, {{.*}} : tensor<4xi64>, tensor<4xi64>) outs({{.*}} : tensor<4xi1>)

  func.func @equal_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xi1> {
    %result = "onnx.Equal"(%a, %b) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
    return %result : tensor<?x?xi1>
  }

  // CHECK-LABEL: func.func @equal_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[A3:.*]]: tensor<?x?xf32>, %[[B3:.*]]: tensor<?x?xf32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty({{.*}}) : tensor<?x?xi1>
  // CHECK: hip.equal(%[[CTX3]]) ins(%[[A3]], %[[B3]] : tensor<?x?xf32>, tensor<?x?xf32>) outs({{.*}} : tensor<?x?xi1>)
}
