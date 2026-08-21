// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(
      %lhs: tensor<2x1x3x1x5x1xf32>,
      %rhs: tensor<1x2x1x4x1x6xf32>)
      -> tensor<2x2x3x4x5x6xf32> {
    // Every adjacent pair mixes a broadcast axis with a non-broadcast axis.
    // Reducing rank six to rank four therefore cannot preserve ONNX
    // per-axis broadcasting.
    // CHECK: error: Mul rank-5/6 shape cannot be packed to rank <= 4 without changing broadcast semantics; all dimensions must be static
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>)
        -> tensor<2x2x3x4x5x6xf32>
    return %result : tensor<2x2x3x4x5x6xf32>
  }
}
