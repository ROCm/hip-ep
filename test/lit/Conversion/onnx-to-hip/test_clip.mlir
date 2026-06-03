// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @clip_static_both_bounds(%x: tensor<3x4xf32>,
                                     %lo: tensor<f32>,
                                     %hi: tensor<f32>) -> tensor<3x4xf32> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<3x4xf32>, tensor<f32>, tensor<f32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @clip_static_both_bounds
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>, %[[LO:.*]]: tensor<f32>, %[[HI:.*]]: tensor<f32>)
  // CHECK-NOT: onnx.Clip
  // CHECK: tensor.empty() : tensor<3x4xf32>
  // CHECK: hip.clip(%[[CTX]]) ins(%[[X]], %[[LO]], %[[HI]] : tensor<3x4xf32>, tensor<f32>, tensor<f32>) outs({{.*}} : tensor<3x4xf32>)

  func.func @clip_only_input(%x: tensor<2x3xf16>) -> tensor<2x3xf16> {
    %y = "onnx.Clip"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
    return %y : tensor<2x3xf16>
  }

  // CHECK-LABEL: func.func @clip_only_input
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.clip

  func.func @clip_dynamic(%x: tensor<?x?xf32>, %lo: tensor<f32>, %hi: tensor<f32>) -> tensor<?x?xf32> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?xf32>
    return %y : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @clip_dynamic
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<?x?xf32>
  // CHECK-NOT: onnx.Clip
  // CHECK: tensor.dim
  // CHECK: tensor.dim
  // CHECK: tensor.empty({{.*}}) : tensor<?x?xf32>
  // CHECK: hip.clip(%[[CTX3]])
}
