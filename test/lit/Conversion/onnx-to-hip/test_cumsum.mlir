// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @cumsum_f32(%x: tensor<3x4xf32>, %axis: tensor<i64>) -> tensor<3x4xf32> {
    %r = "onnx.CumSum"(%x, %axis) : (tensor<3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @cumsum_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>, %[[AX:.*]]: tensor<i64>)
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.cumsum(%[[CTX]]) ins(%[[X]] : tensor<3x4xf32>) axis(%[[AX]] : tensor<i64>) outs({{.*}} : tensor<3x4xf32>)

  func.func @cumsum_attrs(%x: tensor<3x4xf32>, %axis: tensor<i64>) -> tensor<3x4xf32> {
    %r = "onnx.CumSum"(%x, %axis) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @cumsum_attrs
  // CHECK: hip.cumsum({{.*}}) ins({{.*}} : tensor<3x4xf32>) axis({{.*}} : tensor<i64>) outs({{.*}} : tensor<3x4xf32>) {exclusive = 1 : i64, reverse = 1 : i64}

  func.func @cumsum_constant_axis(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %axis = "onnx.Constant"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %r = "onnx.CumSum"(%x, %axis) : (tensor<3x4xf32>, tensor<i64>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @cumsum_constant_axis
  // CHECK-NOT: axis(
  // CHECK: hip.cumsum({{.*}}) ins({{.*}} : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>) {axis_attr = 1 : i64}

  func.func @cumsum_constant_axis_rank1(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %axis = arith.constant dense<[0]> : tensor<1xi32>
    %r = "onnx.CumSum"(%x, %axis) : (tensor<3x4xf32>, tensor<1xi32>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @cumsum_constant_axis_rank1
  // CHECK-NOT: axis(
  // CHECK: hip.cumsum({{.*}}) ins({{.*}} : tensor<3x4xf32>) outs({{.*}} : tensor<3x4xf32>) {axis_attr = 0 : i64}

  func.func @cumsum_dynamic(%x: tensor<?x?xf32>, %axis: tensor<i32>) -> tensor<?x?xf32> {
    %r = "onnx.CumSum"(%x, %axis) : (tensor<?x?xf32>, tensor<i32>) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @cumsum_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<?x?xf32>, %[[AX3:.*]]: tensor<i32>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty
  // CHECK: hip.cumsum(%[[CTX3]]) ins(%[[X3]] : tensor<?x?xf32>) axis(%[[AX3]] : tensor<i32>) outs({{.*}} : tensor<?x?xf32>)
}
