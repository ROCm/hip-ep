// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(
      %lhs: tensor<2x1x3x1x5x1xf32>,
      %rhs: tensor<1x2x1x4x1x6xf32>)
    -> tensor<2x2x3x4x5x6xf32> {
    // Adjacent axes mix broadcast and data, so packing to rank <= 4 would
    // change ONNX per-axis broadcast. Convert hip.mul at the original rank
    // and leave rejection to HIP-to-LLVM.
    // CHECK-LABEL: func.func @main_graph
    // CHECK-NOT: tensor.collapse_shape
    // CHECK: hip.mul({{.*}}) ins({{.*}}, {{.*}} : tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>) outs({{.*}} : tensor<2x2x3x4x5x6xf32>)
    // CHECK-NOT: tensor.expand_shape
    %result = "onnx.Mul"(%lhs, %rhs) :
        (tensor<2x1x3x1x5x1xf32>, tensor<1x2x1x4x1x6xf32>)
        -> tensor<2x2x3x4x5x6xf32>
    return %result : tensor<2x2x3x4x5x6xf32>
  }
}
