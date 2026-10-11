// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(%a: tensor<2x1x4x8xf16>,
                        %b: tensor<1x3x8x16xf16>)
      -> tensor<2x3x4x16xf16> {
    // CHECK: error: matmul partial per-axis batch broadcast is not supported by the strided-batch runtime
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<2x1x4x8xf16>, tensor<1x3x8x16xf16>)
        -> tensor<2x3x4x16xf16>
    return %result : tensor<2x3x4x16xf16>
  }
}
