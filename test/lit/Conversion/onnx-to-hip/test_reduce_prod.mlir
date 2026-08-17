// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    return %arg0 : tensor<4xi32>
  }

  // ReduceProd with axes operand (opset 18+). Both keepdims=1 and
  // noop_with_empty_axes=0 are defaults, so they are elided from attr-dict.
  func.func @reduce_prod_axes_operand(%data: tensor<3x2x2xi32>) -> tensor<3x1x2xi32> {
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.ReduceProd"(%data, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xi32>, tensor<1xi64>) -> tensor<3x1x2xi32>
    return %r : tensor<3x1x2xi32>
  }

  // CHECK-LABEL: func.func @reduce_prod_axes_operand
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<3x2x2xi32>)
  // CHECK: %[[A:.*]] = arith.constant dense<1> : tensor<1xi64>
  // CHECK: tensor.empty() : tensor<3x1x2xi32>
  // CHECK: hip.reduce_prod(%[[CTX]]) ins(%[[D]], %[[A]] : tensor<3x2x2xi32>, tensor<1xi64>) outs({{.*}} : tensor<3x1x2xi32>)

  // ReduceProd with no axes (default: reduce all axes) -> synthesizes a
  // dense<[0, 1, 2]> constant.
  func.func @reduce_prod_default_axes(%data: tensor<3x2x2xi32>) -> tensor<1x1x1xi32> {
    %r = "onnx.ReduceProd"(%data) {keepdims = 1 : si64} : (tensor<3x2x2xi32>) -> tensor<1x1x1xi32>
    return %r : tensor<1x1x1xi32>
  }

  // CHECK-LABEL: func.func @reduce_prod_default_axes
  // CHECK: arith.constant dense<[0, 1, 2]> : tensor<3xi64>
  // CHECK: hip.reduce_prod({{.*}}) ins({{.*}}, {{.*}} : tensor<3x2x2xi32>, tensor<3xi64>) outs({{.*}} : tensor<1x1x1xi32>)

  // ReduceProd with axes attribute (older opset).
  func.func @reduce_prod_axes_attr(%data: tensor<3x2x2xi32>) -> tensor<3x1x2xi32> {
    %r = "onnx.ReduceProd"(%data) {axes = [1 : si64], keepdims = 1 : si64} : (tensor<3x2x2xi32>) -> tensor<3x1x2xi32>
    return %r : tensor<3x1x2xi32>
  }

  // CHECK-LABEL: func.func @reduce_prod_axes_attr
  // CHECK: arith.constant dense<1> : tensor<1xi64>
  // CHECK: hip.reduce_prod({{.*}}) ins({{.*}}, {{.*}} : tensor<3x2x2xi32>, tensor<1xi64>) outs({{.*}} : tensor<3x1x2xi32>)

  // Dynamic input with keepdims=1: axis 1 is reduced -> output dim
  // for that axis is the constant 1; the non-reduced dims (0 and 2)
  // forward through via tensor.dim where input is dynamic.
  func.func @reduce_prod_dyn_keepdims(%data: tensor<?x2x?xi32>) -> tensor<?x1x?xi32> {
    %r = "onnx.ReduceProd"(%data) {axes = [1 : si64], keepdims = 1 : si64} : (tensor<?x2x?xi32>) -> tensor<?x1x?xi32>
    return %r : tensor<?x1x?xi32>
  }

  // CHECK-LABEL: func.func @reduce_prod_dyn_keepdims
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x2x?xi32>)
  // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[D]], %[[A0]] : tensor<?x2x?xi32>
  // CHECK-DAG: %[[A2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[D2:.*]] = tensor.dim %[[D]], %[[A2]] : tensor<?x2x?xi32>
  // CHECK: tensor.empty(%[[D0]], %[[D2]]) : tensor<?x1x?xi32>
  // CHECK: hip.reduce_prod

  // Dynamic input with keepdims=0: axis 1 is reduced and dropped from
  // the output. Output rank = 2; output[0] <- input[0], output[1] <-
  // input[2] (the outIdx -> inIdx mapping skips the reduced axis).
  func.func @reduce_prod_dyn_drop_dim(%data: tensor<?x2x?xi32>) -> tensor<?x?xi32> {
    %r = "onnx.ReduceProd"(%data) {axes = [1 : si64], keepdims = 0 : si64} : (tensor<?x2x?xi32>) -> tensor<?x?xi32>
    return %r : tensor<?x?xi32>
  }

  // CHECK-LABEL: func.func @reduce_prod_dyn_drop_dim
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x2x?xi32>)
  // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[D]], %[[A0]] : tensor<?x2x?xi32>
  // CHECK-DAG: %[[A2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[D2:.*]] = tensor.dim %[[D]], %[[A2]] : tensor<?x2x?xi32>
  // CHECK: tensor.empty(%[[D0]], %[[D2]]) : tensor<?x?xi32>
  // CHECK: hip.reduce_prod({{.*}}) ins({{.*}}, {{.*}} : tensor<?x2x?xi32>, tensor<1xi64>) outs({{.*}} : tensor<?x?xi32>) {keepdims = 0 : i64, normalized_axes = array<i64: 1>
}
