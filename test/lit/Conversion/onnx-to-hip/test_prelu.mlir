// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Per-channel slope, the ArcFace form: 1xCx1x1 onto 1xCxHxW.
  func.func @prelu_per_channel(%x: tensor<1x64x8x8xf32>,
                               %slope: tensor<1x64x1x1xf32>) -> tensor<1x64x8x8xf32> {
    // CHECK-LABEL: func.func @prelu_per_channel
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x64x8x8xf32>, %[[SLOPE:.*]]: tensor<1x64x1x1xf32>)
    %y = "onnx.PRelu"(%x, %slope)
        : (tensor<1x64x8x8xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x8x8xf32>
    // CHECK-NOT: onnx.PRelu
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x64x8x8xf32>
    // CHECK: hip.prelu(%[[CTX]]) ins(%[[X]], %[[SLOPE]] : tensor<1x64x8x8xf32>, tensor<1x64x1x1xf32>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x64x8x8xf32>)
    return %y : tensor<1x64x8x8xf32>
  }

  // Scalar slope broadcasts onto every element.
  func.func @prelu_scalar_slope(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
    // CHECK-LABEL: func.func @prelu_scalar_slope
    %slope = "onnx.Constant"() {value = dense<1.000000e-01> : tensor<f16>} : () -> tensor<f16>
    %y = "onnx.PRelu"(%x, %slope) : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x3xf16>
    // CHECK-NOT: onnx.PRelu
    // CHECK: hip.prelu
    // CHECK-SAME: tensor<f16>
    return %y : tensor<2x3xf16>
  }

  // Rank-1 slope aligns to the trailing dimension.
  func.func @prelu_trailing_slope(%x: tensor<2x4xf32>, %slope: tensor<4xf32>)
      -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @prelu_trailing_slope
    %y = "onnx.PRelu"(%x, %slope) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    // CHECK-NOT: onnx.PRelu
    // CHECK: hip.prelu
    return %y : tensor<2x4xf32>
  }

  // A slope that does not broadcast onto X stays unconverted.
  func.func @prelu_slope_mismatch(%x: tensor<1x2x4x4xf32>, %slope: tensor<3xf32>)
      -> tensor<1x2x4x4xf32> {
    // CHECK-LABEL: func.func @prelu_slope_mismatch
    %y = "onnx.PRelu"(%x, %slope)
        : (tensor<1x2x4x4xf32>, tensor<3xf32>) -> tensor<1x2x4x4xf32>
    // CHECK: onnx.PRelu
    // CHECK-NOT: hip.prelu
    return %y : tensor<1x2x4x4xf32>
  }
}
