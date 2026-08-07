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

  func.func @expand_constant_shape(%input: tensor<1x3x1xf32>) -> tensor<?x3x?xf32> {
    %shape = arith.constant dense<[2, 3, 6]> : tensor<3xi64>
    %r = "onnx.Expand"(%input, %shape) : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<?x3x?xf32>
    return %r : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @expand_constant_shape
  // CHECK: %[[D0:.*]] = arith.constant 2 : index
  // CHECK: %[[D2:.*]] = arith.constant 6 : index
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.empty(%[[D0]], %[[D2]]) : tensor<?x3x?xf32>
  // CHECK: hip.expand
}
