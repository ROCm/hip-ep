// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(%a: tensor<2x4xf16>, %b: tensor<4x5xf16>,
                        %c: tensor<3xf16>) -> tensor<2x5xf16> {
    // CHECK: error: gemm C dimension 3 is not broadcastable to output dimension 5
    %result = "onnx.Gemm"(%a, %b, %c)
      : (tensor<2x4xf16>, tensor<4x5xf16>, tensor<3xf16>)
        -> tensor<2x5xf16>
    return %result : tensor<2x5xf16>
  }
}
