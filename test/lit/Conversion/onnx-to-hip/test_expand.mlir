// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @expand_static(%input: tensor<3x1xf32>, %shape: tensor<3xi64>) -> tensor<2x3x6xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<2x3x6xf32>
    return %r : tensor<2x3x6xf32>
  }

  // CHECK-LABEL: func.func @expand_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<3x1xf32>, %[[SH:.*]]: tensor<3xi64>)
  // CHECK: tensor.empty() : tensor<2x3x6xf32>
  // CHECK: hip.expand(%[[CTX]]) ins(%[[IN]], %[[SH]] : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<2x3x6xf32>)

  func.func @expand_dynamic(%input: tensor<3x1xf32>, %shape: tensor<3xi64>) -> tensor<?x3x?xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<?x3x?xf32>
    return %r : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @expand_dynamic
  // CHECK: tensor.extract
  // CHECK: tensor.extract
  // CHECK: tensor.empty
  // CHECK: hip.expand({{.*}}) ins({{.*}}, {{.*}} : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<?x3x?xf32>)

  // Opaque shape operand with a possibly-> 1 input dim: honour ONNX Expand
  // broadcast rule output_dim = max(input_dim, shape_value) so an
  // identity-broadcast shape (e.g. [1, ...]) does not shrink the input.
  func.func @expand_maxsi(%input: tensor<?x4xf32>, %shape: tensor<2xi64>) -> tensor<?x4xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<?x4xf32>
    return %r : tensor<?x4xf32>
  }

  // CHECK-LABEL: func.func @expand_maxsi
  // CHECK: tensor.extract
  // CHECK: tensor.dim
  // CHECK: arith.maxsi
  // CHECK: hip.expand({{.*}}) ins({{.*}}, {{.*}} : tensor<?x4xf32>, tensor<2xi64>) outs({{.*}} : tensor<?x4xf32>)
}
