// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// onnx.Clip decomposes into onnx.Max(x, lo) followed by onnx.Min(_, hi),
// each of which is then lowered to hip.max / hip.min via miopenOpTensor.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @clip_both_bounds(%x: tensor<3x4xf32>, %lo: tensor<f32>, %hi: tensor<f32>) -> tensor<3x4xf32> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<3x4xf32>, tensor<f32>, tensor<f32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @clip_both_bounds
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<3x4xf32>, %[[LO:.*]]: tensor<f32>, %[[HI:.*]]: tensor<f32>)
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max(%[[CTX]]) ins(%[[X]], %[[LO]]
  // CHECK: hip.min(%[[CTX]])

  // Only lower bound supplied: produces just hip.max.
  func.func @clip_only_lo(%x: tensor<3x4xf32>, %lo: tensor<f32>) -> tensor<3x4xf32> {
    %y = "onnx.Clip"(%x, %lo) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
    return %y : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @clip_only_lo
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max
  // CHECK-NOT: hip.min

  // Dynamic shape.
  func.func @clip_dynamic(%x: tensor<?x?xf16>, %lo: tensor<f16>, %hi: tensor<f16>) -> tensor<?x?xf16> {
    %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<?x?xf16>, tensor<f16>, tensor<f16>) -> tensor<?x?xf16>
    return %y : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @clip_dynamic
  // CHECK-NOT: onnx.Clip
  // CHECK: hip.max
  // CHECK: hip.min

  // Rank-5 input, packed to the 4-D path before the max/min decomposition.
  // The scalar bounds are forwarded unchanged. Shape taken from bevformer's
  // Clip_1113, which the elementwise ABI rejects above rank 4.
  func.func @clip_5d_scalar_bounds(%x: tensor<4x1x6x2500x1xf32>, %lo: tensor<f32>)
      -> tensor<4x1x6x2500x1xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Clip"(%x, %lo, %none) : (tensor<4x1x6x2500x1xf32>, tensor<f32>, none) -> tensor<4x1x6x2500x1xf32>
    return %y : tensor<4x1x6x2500x1xf32>
  }

  // CHECK-LABEL: func.func @clip_5d_scalar_bounds
  // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[X5:.*]]: tensor<4x1x6x2500x1xf32>, %[[LO5:.*]]: tensor<f32>)
  // CHECK-NOT: onnx.Clip
  // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<4x1x6x2500x1xf32> into tensor<4x1x6x2500xf32>
  // CHECK: hip.max(%[[CTX5]]) ins({{.*}}, {{.*}} : tensor<4x1x6x2500xf32>, tensor<f32>) outs({{.*}} : tensor<4x1x6x2500xf32>)
  // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 1, 6, 2500, 1] : tensor<4x1x6x2500xf32> into tensor<4x1x6x2500x1xf32>

  // A rank-5 Clip with both bounds absent is an identity, so it is left alone
  // rather than packed: there is no hip.max/hip.min to keep under the ceiling.
  func.func @clip_5d_no_bounds(%x: tensor<4x1x6x2500x1xf32>) -> tensor<4x1x6x2500x1xf32> {
    %n1 = "onnx.NoValue"() {value} : () -> none
    %n2 = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Clip"(%x, %n1, %n2) : (tensor<4x1x6x2500x1xf32>, none, none) -> tensor<4x1x6x2500x1xf32>
    return %y : tensor<4x1x6x2500x1xf32>
  }

  // CHECK-LABEL: func.func @clip_5d_no_bounds
  // CHECK-SAME: (%[[CTXN:.*]]: !hip.context, %[[XN:.*]]: tensor<4x1x6x2500x1xf32>)
  // CHECK-NOT: tensor.collapse_shape
  // CHECK-NOT: hip.max
  // CHECK: return %[[XN]] : tensor<4x1x6x2500x1xf32>

  // Rank-5 relu: same ceiling, reached through hip.max(x, 0).
  func.func @relu_5d(%x: tensor<4x1x6x2500x1xf32>) -> tensor<4x1x6x2500x1xf32> {
    %y = "onnx.Relu"(%x) : (tensor<4x1x6x2500x1xf32>) -> tensor<4x1x6x2500x1xf32>
    return %y : tensor<4x1x6x2500x1xf32>
  }

  // CHECK-LABEL: func.func @relu_5d
  // CHECK-NOT: onnx.Relu
  // CHECK: tensor.collapse_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] : tensor<4x1x6x2500x1xf32> into tensor<4x1x6x2500xf32>
  // CHECK: hip.max
  // CHECK: tensor.expand_shape {{.*}} {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 1, 6, 2500, 1] : tensor<4x1x6x2500xf32> into tensor<4x1x6x2500x1xf32>
}
