// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<6x4xf16>) -> tensor<6x4xf16> {
    return %arg0 : tensor<6x4xf16>
  }

  func.func @varlen_conv_silu(
      %input: tensor<6x4xf16>,
      %weight: tensor<4x1x4xf16>,
      %cumulative: tensor<3xi32>,
      %bias: tensor<4xf16>,
      %state: tensor<2x4x3xf16>
  ) -> (tensor<6x4xf16>, tensor<2x4x3xf16>) {
    %output, %present = "onnx.Custom"(%input, %weight, %cumulative, %bias, %state) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      dilation = 1 : si64
    } : (tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, tensor<4xf16>, tensor<2x4x3xf16>)
      -> (tensor<6x4xf16>, tensor<2x4x3xf16>)
    return %output, %present : tensor<6x4xf16>, tensor<2x4x3xf16>
  }

  // CHECK-LABEL: func.func @varlen_conv_silu
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<6x4xf16>, %[[W:.*]]: tensor<4x1x4xf16>, %[[CU:.*]]: tensor<3xi32>, %[[B:.*]]: tensor<4xf16>, %[[ST:.*]]: tensor<2x4x3xf16>)
  // CHECK: %[[OUT:.*]] = tensor.empty() : tensor<6x4xf16>
  // CHECK: hip.varlen_causal_conv_with_state(%[[CTX]]) ins(%[[IN]], %[[W]], %[[CU]], %[[B]], %[[ST]] : tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, tensor<4xf16>, tensor<2x4x3xf16>) outs(%[[OUT]], %[[ST]] : tensor<6x4xf16>, tensor<2x4x3xf16>) {activation = "silu"}

  func.func @varlen_conv_capacity(
      %input: tensor<6x4xf16>,
      %weight: tensor<4x1x4xf16>,
      %cumulative: tensor<3xi32>,
      %bias: tensor<4xf16>,
      %state: tensor<2x4x3xf16>
  ) -> (tensor<6x4xf16>, tensor<2x4x3xf16>) {
    %output, %present = "onnx.Custom"(%input, %weight, %cumulative, %bias, %state) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      state_update_capacity = 2 : si64
    } : (tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, tensor<4xf16>, tensor<2x4x3xf16>)
      -> (tensor<6x4xf16>, tensor<2x4x3xf16>)
    return %output, %present : tensor<6x4xf16>, tensor<2x4x3xf16>
  }

  // CHECK-LABEL: func.func @varlen_conv_capacity
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.varlen_causal_conv_with_state
}
