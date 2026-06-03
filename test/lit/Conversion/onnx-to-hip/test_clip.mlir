// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// onnx.Clip decomposes into onnx.Max(x, lo) followed by onnx.Min(_, hi),
// each of which is then lowered to hip.max / hip.min via miopenOpTensor.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @clip_both_bounds(%x: tensor<3x4xf32>, %lo: tensor<f32>, %hi: tensor<f32>) -> tensor<3x4xf32> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<3x4xf32>, tensor<f32>, tensor<f32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @clip_both_bounds
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>, %[[LO:.*]]: tensor<f32>, %[[HI:.*]]: tensor<f32>)
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max(%[[CTX]]) ins(%[[X]], %[[LO]]
  // CHECK: hip.min(%[[CTX]])

  // Only lower bound supplied: produces just hip.max.
  func.func @clip_only_lo(%x: tensor<3x4xf32>, %lo: tensor<f32>) -> tensor<3x4xf32> {
    %y = "onnx.Clip"(%x, %lo) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @clip_only_lo
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max
  // CHECK-NOT: hip.min

  // Dynamic shape.
  func.func @clip_dynamic(%x: tensor<?x?xf16>, %lo: tensor<f16>, %hi: tensor<f16>) -> tensor<?x?xf16> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<?x?xf16>, tensor<f16>, tensor<f16>) -> tensor<?x?xf16>
    return %y : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @clip_dynamic
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max
  // CHECK: hip.min
}
