// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Constant pad (default mode), no constant_value, no axes.
  // Default mode is elided from attr-dict, so just check the op shape.
  func.func @pad_constant_default(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_default
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<3x4xf32>, %[[P:.*]]: tensor<4xi64>)
  // CHECK: tensor.empty() : tensor<5x6xf32>
  // CHECK: hip.pad(%[[CTX]]) ins(%[[D]], %[[P]] : tensor<3x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<5x6xf32>)

  // Constant pad with explicit constant_value scalar.
  func.func @pad_constant_with_cval(%data: tensor<3x4xf32>, %pads: tensor<4xi64>, %cval: tensor<f32>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %cval, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_with_cval
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) cval({{.*}} : tensor<f32>) outs({{.*}} : tensor<5x6xf32>)

  // Reflect mode is non-default, so it stays in the attr-dict.
  func.func @pad_reflect(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "reflect"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_reflect
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<5x6xf32>) {mode = "reflect"}

  func.func @pad_edge(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "edge"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_edge
  // CHECK: hip.pad({{.*}}) {{.*}} {mode = "edge"}

  // With axes input.
  func.func @pad_axes(%data: tensor<3x4xf32>, %pads: tensor<2xi64>, %axes: tensor<1xi64>) -> tensor<3x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %axes) {mode = "constant"} : (tensor<3x4xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<3x6xf32>
    return %r : tensor<3x6xf32>
  }

  // CHECK-LABEL: func.func @pad_axes
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<2xi64>) axes({{.*}} : tensor<1xi64>) outs({{.*}} : tensor<3x6xf32>)
}
