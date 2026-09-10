// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @swish_default(%x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %y = "onnx.Swish"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @swish_default
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>)
  // CHECK-NOT: onnx.Swish
  // CHECK: hip.swish(%[[CTX]]) ins(%[[X]] : tensor<3x4xf32>)
  // CHECK-NOT: alpha
  // CHECK-SAME: : tensor<3x4xf32>

  func.func @swish_alpha_f16(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
    %y = "onnx.Swish"(%x) {alpha = 0.5 : f32}
        : (tensor<2x3xf16>) -> tensor<2x3xf16>
    return %y : tensor<2x3xf16>
  }

  // CHECK-LABEL: func.func @swish_alpha_f16
  // CHECK-NOT: onnx.Swish
  // CHECK: hip.swish
  // CHECK-SAME: {alpha = 5.000000e-01 : f64}

  func.func @swish_bf16(%x: tensor<2x5xbf16>) -> tensor<2x5xbf16> {
    %y = "onnx.Swish"(%x) {alpha = 2.0 : f32}
        : (tensor<2x5xbf16>) -> tensor<2x5xbf16>
    return %y : tensor<2x5xbf16>
  }

  // CHECK-LABEL: func.func @swish_bf16
  // CHECK-NOT: onnx.Swish
  // CHECK: hip.swish

  func.func @swish_dynamic_f64(%x: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %y = "onnx.Swish"(%x) {alpha = -0.25 : f32}
        : (tensor<?x?xf64>) -> tensor<?x?xf64>
    return %y : tensor<?x?xf64>
  }

  // CHECK-LABEL: func.func @swish_dynamic_f64
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[X2:.*]]: tensor<?x?xf64>)
  // CHECK-NOT: onnx.Swish
  // CHECK: tensor.dim
  // CHECK: hip.swish(%[[CTX2]])
  // CHECK-SAME: {alpha = -2.500000e-01 : f64}
}
