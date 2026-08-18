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
  // CHECK-NOT: hip.readback_scalar
  // CHECK-COUNT-1: hip.readback_control
  // CHECK-COUNT-3: hip.checked_expand_extent
  // CHECK: tensor.empty() : tensor<2x3x6xf32>
  // CHECK: hip.expand(%[[CTX]]) valid({{.*}}) ins(%[[IN]], %[[SH]] : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<2x3x6xf32>)

  func.func @expand_dynamic(%input: tensor<3x1xf32>, %shape: tensor<3xi64>) -> tensor<?x3x?xf32> {
    %r = "onnx.Expand"(%input, %shape) : (tensor<3x1xf32>, tensor<3xi64>) -> tensor<?x3x?xf32>
    return %r : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @expand_dynamic
  // CHECK-NOT: hip.readback_scalar
  // CHECK-COUNT-1: hip.readback_control
  // CHECK-COUNT-3: hip.checked_expand_extent
  // CHECK: tensor.empty
  // CHECK: hip.expand({{.*}}) valid({{.*}}) ins({{.*}}, {{.*}} : tensor<3x1xf32>, tensor<3xi64>) outs({{.*}} : tensor<?x3x?xf32>)

  func.func @expand_constant_shape(%input: tensor<1x3x1xf32>) -> tensor<?x3x?xf32> {
    %shape = "onnx.Constant"() {
      value = dense<[2, 3, 6]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %r = "onnx.Expand"(%input, %shape) : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<?x3x?xf32>
    return %r : tensor<?x3x?xf32>
  }

  // CHECK-LABEL: func.func @expand_constant_shape
  // CHECK-DAG: %[[SHAPE:.*]] = hip.constant {{.*}}value = dense<[2, 3, 6]>
  // CHECK-DAG: %[[D0:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[D2:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[VALID:.*]] = arith.constant true
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.empty(%[[D0]], %[[D2]]) : tensor<?x3x?xf32>
  // CHECK: hip.expand({{.*}}) valid(%[[VALID]]) ins({{.*}}, %[[SHAPE]]

  func.func @expand_constant_one_dynamic_input(
      %input: tensor<?xf32>) -> tensor<?xf32> {
    %shape = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Expand"(%input, %shape)
      : (tensor<?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @expand_constant_one_dynamic_input
  // CHECK-NOT: hip.readback_control
  // CHECK: tensor.dim
  // CHECK: hip.checked_expand_extent
  // CHECK: tensor.empty
  // CHECK: hip.expand({{.*}}) valid({{.*}})

  func.func @expand_runtime_target(
      %input: tensor<?xf32>, %shape: tensor<1xi64>) -> tensor<?xf32> {
    %r = "onnx.Expand"(%input, %shape)
      : (tensor<?xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }

  // CHECK-LABEL: func.func @expand_runtime_target
  // CHECK-NOT: hip.readback_scalar
  // CHECK-COUNT-1: hip.readback_control
  // CHECK: tensor.dim
  // CHECK-COUNT-1: hip.checked_expand_extent
  // CHECK: hip.expand({{.*}}) valid({{.*}})

  func.func @expand_zero_input(%input: tensor<0xf32>) -> tensor<0xf32> {
    %shape = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Expand"(%input, %shape)
      : (tensor<0xf32>, tensor<1xi64>) -> tensor<0xf32>
    return %r : tensor<0xf32>
  }

  // CHECK-LABEL: func.func @expand_zero_input
  // CHECK: tensor.empty() : tensor<0xf32>
  // CHECK: hip.expand({{.*}}) valid({{.*}})

  func.func @expand_zero_target(%input: tensor<1xf32>) -> tensor<0xf32> {
    %shape = arith.constant dense<[0]> : tensor<1xi64>
    %r = "onnx.Expand"(%input, %shape)
      : (tensor<1xf32>, tensor<1xi64>) -> tensor<0xf32>
    return %r : tensor<0xf32>
  }

  // CHECK-LABEL: func.func @expand_zero_target
  // CHECK: tensor.empty() : tensor<0xf32>
  // CHECK: hip.expand({{.*}}) valid({{.*}})

  func.func @expand_rank_zero(%input: tensor<f32>) -> tensor<f32> {
    %shape = arith.constant dense<> : tensor<0xi64>
    %r = "onnx.Expand"(%input, %shape)
      : (tensor<f32>, tensor<0xi64>) -> tensor<f32>
    return %r : tensor<f32>
  }

  // CHECK-LABEL: func.func @expand_rank_zero
  // CHECK: tensor.empty() : tensor<f32>
  // CHECK: hip.expand({{.*}}) valid({{.*}})
}
