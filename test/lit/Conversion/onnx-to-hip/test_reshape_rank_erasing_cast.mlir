// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1xf16>) -> tensor<1xf16> {
    return %input : tensor<1xf16>
  }

  func.func @ranked_cast_reshape(
      %data: tensor<2x3x4xf16>,
      %shape: tensor<3xi64>) -> tensor<2x6x2xf16> {
    %unranked = tensor.cast %data : tensor<2x3x4xf16> to tensor<*xf16>
    %result = "onnx.Reshape"(%unranked, %shape)
        : (tensor<*xf16>, tensor<3xi64>) -> tensor<2x6x2xf16>
    return %result : tensor<2x6x2xf16>
  }

  // A genuinely unranked block argument has no ranked source to recover. The
  // failed match must leave the original operation untouched and emit no
  // speculative reshape IR.
  func.func @genuinely_unranked_reshape(
      %data: tensor<*xf16>,
      %shape: tensor<3xi64>) -> tensor<2x6x2xf16> {
    %result = "onnx.Reshape"(%data, %shape)
        : (tensor<*xf16>, tensor<3xi64>) -> tensor<2x6x2xf16>
    return %result : tensor<2x6x2xf16>
  }
}

// CHECK-LABEL: func.func @ranked_cast_reshape(
// CHECK-SAME: %{{[^:]+}}: !hip.context, %[[DATA:[^:]+]]: tensor<2x3x4xf16>
// CHECK-NOT: tensor.cast
// CHECK: tensor.collapse_shape %[[DATA]]
// CHECK: tensor.expand_shape
// CHECK-NOT: "onnx.Reshape"

// CHECK-LABEL: func.func @genuinely_unranked_reshape(
// CHECK-SAME: %{{[^:]+}}: !hip.context,
// CHECK-SAME: %[[UNRANKED_DATA:[^:]+]]: tensor<*xf16>,
// CHECK-SAME: %[[SHAPE:[^:]+]]: tensor<3xi64>
// CHECK-NOT: tensor.dim
// CHECK-NOT: tensor.collapse_shape
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: tensor.reshape
// CHECK-NOT: arith.
// CHECK: "onnx.Reshape"(%[[UNRANKED_DATA]], %[[SHAPE]])
