// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Residual add from opset-7 ResNet-50: two equal rank-4 tensors.
  func.func @sum_residual(%a: tensor<1x256x56x56xf32>, %b: tensor<1x256x56x56xf32>)
      -> tensor<1x256x56x56xf32> {
    // CHECK-LABEL: func.func @sum_residual
    %y = "onnx.Sum"(%a, %b)
        : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    // CHECK-NOT: onnx.Sum
    // CHECK: hip.add
    // CHECK-SAME: tensor<1x256x56x56xf32>
    return %y : tensor<1x256x56x56xf32>
  }

  // A shorter-rank operand broadcasts onto the result.
  func.func @sum_broadcast(%a: tensor<1x4x8x8xf16>, %b: tensor<1x4x1x1xf16>)
      -> tensor<1x4x8x8xf16> {
    // CHECK-LABEL: func.func @sum_broadcast
    %y = "onnx.Sum"(%a, %b)
        : (tensor<1x4x8x8xf16>, tensor<1x4x1x1xf16>) -> tensor<1x4x8x8xf16>
    // CHECK-NOT: onnx.Sum
    // CHECK: hip.add
    // CHECK-SAME: tensor<1x4x1x1xf16>
    return %y : tensor<1x4x8x8xf16>
  }

  // Three inputs fold left into two adds.
  func.func @sum_three(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>, %c: tensor<4xf32>)
      -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @sum_three
    %y = "onnx.Sum"(%a, %b, %c)
        : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    // CHECK-NOT: onnx.Sum
    // CHECK: hip.add
    // CHECK: hip.add
    return %y : tensor<2x4xf32>
  }

  // Rank 5 is packed to rank 4, then added, then expanded.
  func.func @sum_rank5(%a: tensor<1x2x3x4x5xf32>, %b: tensor<1x2x3x4x5xf32>)
      -> tensor<1x2x3x4x5xf32> {
    // CHECK-LABEL: func.func @sum_rank5
    %y = "onnx.Sum"(%a, %b)
        : (tensor<1x2x3x4x5xf32>, tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
    // CHECK-NOT: onnx.Sum
    // CHECK: tensor.collapse_shape
    // CHECK: hip.add
    // CHECK: tensor.expand_shape
    return %y : tensor<1x2x3x4x5xf32>
  }

  // Incompatible shapes stay unconverted.
  func.func @sum_mismatch(%a: tensor<2x3xf32>, %b: tensor<4xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @sum_mismatch
    %y = "onnx.Sum"(%a, %b) : (tensor<2x3xf32>, tensor<4xf32>) -> tensor<2x3xf32>
    // CHECK: onnx.Sum
    // CHECK-NOT: hip.add
    return %y : tensor<2x3xf32>
  }
}
