// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A failed payload rewrite must leave no tensor.dim/arithmetic/empty/readback
// operations behind. Builds with MLIR expensive pattern API checks also reject
// any mutation followed by match failure.
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1xf32>) -> tensor<1xf32> {
    return %input : tensor<1xf32>
  }

  func.func @pad_static_contradiction(
      %input: tensor<3x4xf32>) -> tensor<5x7xf32> {
    %pads = "onnx.Constant"() {
      value = dense<[1, 1, 1, 1]> : tensor<4xi64>
    } : () -> tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Pad"(%input, %pads, %none, %none) {mode = "constant"}
      : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x7xf32>
    return %result : tensor<5x7xf32>
  }

  // CHECK-LABEL: func.func @pad_static_contradiction
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: hip.pad
  // CHECK: "onnx.Pad"

  func.func @tile_static_contradiction(
      %input: tensor<2x3xf32>) -> tensor<4x8xf32> {
    %repeats = "onnx.Constant"() {
      value = dense<[2, 3]> : tensor<2xi64>
    } : () -> tensor<2xi64>
    %result = "onnx.Tile"(%input, %repeats)
      : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }

  // CHECK-LABEL: func.func @tile_static_contradiction
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: hip.tile
  // CHECK: "onnx.Tile"

  func.func @expand_static_contradiction(
      %input: tensor<1x3x1xf32>) -> tensor<2x3x5xf32> {
    %shape = "onnx.Constant"() {
      value = dense<[2, 3, 6]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    %result = "onnx.Expand"(%input, %shape)
      : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<2x3x5xf32>
    return %result : tensor<2x3x5xf32>
  }

  // CHECK-LABEL: func.func @expand_static_contradiction
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: hip.readback
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: hip.expand
  // CHECK: "onnx.Expand"

  // Static imported dimensions may refine dynamic payload inference.
  func.func @pad_static_result_refinement(
      %input: tensor<?x4xf32>) -> tensor<5x4xf32> {
    %pads = arith.constant dense<[1, 0, 1, 0]> : tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Pad"(%input, %pads, %none, %none) {mode = "constant"}
      : (tensor<?x4xf32>, tensor<4xi64>, none, none) -> tensor<5x4xf32>
    return %result : tensor<5x4xf32>
  }

  // CHECK-LABEL: func.func @pad_static_result_refinement
  // CHECK: tensor.empty() : tensor<5x4xf32>
  // CHECK: hip.pad

  func.func @tile_static_result_refinement(
      %input: tensor<?x3xf32>) -> tensor<4x9xf32> {
    %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
    %result = "onnx.Tile"(%input, %repeats)
      : (tensor<?x3xf32>, tensor<2xi64>) -> tensor<4x9xf32>
    return %result : tensor<4x9xf32>
  }

  // CHECK-LABEL: func.func @tile_static_result_refinement
  // CHECK: tensor.empty() : tensor<4x9xf32>
  // CHECK: hip.tile

  func.func @expand_static_result_refinement(
      %input: tensor<?x1xf32>) -> tensor<3x4xf32> {
    %shape = arith.constant dense<[1, 4]> : tensor<2xi64>
    %result = "onnx.Expand"(%input, %shape)
      : (tensor<?x1xf32>, tensor<2xi64>) -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @expand_static_result_refinement
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.expand

  // Dynamic imported dimensions may be refined by proven static inference.
  func.func @pad_dynamic_result_static_inference(
      %input: tensor<3x4xf32>) -> tensor<?x?xf32> {
    %pads = arith.constant dense<[1, 1, 1, 1]> : tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Pad"(%input, %pads, %none, %none) {mode = "constant"}
      : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @pad_dynamic_result_static_inference
  // CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf32>
  // CHECK: hip.pad

  func.func @tile_dynamic_result_static_inference(
      %input: tensor<2x3xf32>) -> tensor<?x?xf32> {
    %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
    %result = "onnx.Tile"(%input, %repeats)
      : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @tile_dynamic_result_static_inference
  // CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf32>
  // CHECK: hip.tile

  func.func @expand_dynamic_result_static_inference(
      %input: tensor<1x3x1xf32>) -> tensor<?x?x?xf32> {
    %shape = arith.constant dense<[2, 3, 6]> : tensor<3xi64>
    %result = "onnx.Expand"(%input, %shape)
      : (tensor<1x3x1xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    return %result : tensor<?x?x?xf32>
  }

  // CHECK-LABEL: func.func @expand_dynamic_result_static_inference
  // CHECK: tensor.empty({{.*}}, {{.*}}, {{.*}}) : tensor<?x?x?xf32>
  // CHECK: hip.expand
}
