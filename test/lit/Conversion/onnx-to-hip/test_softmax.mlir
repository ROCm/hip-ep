// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
    return %arg0 : tensor<2x4xbf16>
  }

  // CHECK-LABEL: func.func @softmax_bf16
  // CHECK: %[[OUT:.*]] = tensor.empty() : tensor<2x4xbf16>
  // CHECK: hip.miopen.softmax(
  // CHECK-SAME: ins(%{{[A-Za-z0-9._]+}} : tensor<2x4xbf16>)
  // CHECK-SAME: outs(%[[OUT]] : tensor<2x4xbf16>)
  func.func @softmax_bf16(%input: tensor<2x4xbf16>) -> tensor<2x4xbf16> {
    %result = "onnx.Softmax"(%input) {axis = -1 : si64}
      : (tensor<2x4xbf16>) -> tensor<2x4xbf16>
    return %result : tensor<2x4xbf16>
  }
}
