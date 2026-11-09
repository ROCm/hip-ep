// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    return %arg0 : tensor<4xf64>
  }

  func.func @isinf_f64(%input: tensor<3x4xf64>) -> tensor<3x4xi1> {
    %result = "onnx.IsInf"(%input) : (tensor<3x4xf64>) -> tensor<3x4xi1>
    return %result : tensor<3x4xi1>
  }

  // CHECK-LABEL: func.func @isinf_f64
  // CHECK: hip.isinf(%{{.*}}) ins(%{{.*}} : tensor<3x4xf64>) outs({{.*}} : tensor<3x4xi1>)
}
