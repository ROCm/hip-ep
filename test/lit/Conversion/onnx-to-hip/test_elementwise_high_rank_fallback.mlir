// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(
      %lhs: tensor<2x1x3x1x5x1xf32>,
      %rhs: tensor<1x2x1x4x1x6xf32>)
      -> tensor<2x2x3x4x5x6xf32> {
    // Every adjacent pair mixes a broadcast axis with a non-broadcast axis,
    // so reducing rank six to rank four cannot preserve ONNX broadcasting.
    // Preserve the rank-6 HIP op for a backend with native support. The current
    // HIP-to-LLVM backend rejects it separately.
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: hip.mul({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>) outs({{.*}} : tensor<2x2x3x4x5x6xf32>)
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>)
        -> tensor<2x2x3x4x5x6xf32>
    return %result : tensor<2x2x3x4x5x6xf32>
  }
}
