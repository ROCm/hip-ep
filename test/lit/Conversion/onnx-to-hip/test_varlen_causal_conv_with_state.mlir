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
  // CHECK: %[[FINAL:.*]] = tensor.empty() : tensor<2x4x3xf16>
  // CHECK: hip.varlen_causal_conv_with_state(%[[CTX]]) ins(%[[IN]], %[[W]], %[[CU]], %[[ST]] : tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, tensor<2x4x3xf16>) bias(%[[B]] : tensor<4xf16>) outs(%[[OUT]], %[[FINAL]] : tensor<6x4xf16>, tensor<2x4x3xf16>) {activation = "silu"}

  func.func @varlen_conv_no_bias(
      %input: tensor<6x4xf16>,
      %weight: tensor<4x1x4xf16>,
      %cumulative: tensor<3xi32>,
      %state: tensor<2x4x3xf16>
  ) -> (tensor<6x4xf16>, tensor<2x4x3xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %final = "onnx.Custom"(%input, %weight, %cumulative, %none, %state) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft"
    } : (tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, none, tensor<2x4x3xf16>)
      -> (tensor<6x4xf16>, tensor<2x4x3xf16>)
    return %output, %final : tensor<6x4xf16>, tensor<2x4x3xf16>
  }

  // CHECK-LABEL: func.func @varlen_conv_no_bias
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[IN2:.*]]: tensor<6x4xf16>, %[[W2:.*]]: tensor<4x1x4xf16>, %[[CU2:.*]]: tensor<3xi32>, %[[ST2:.*]]: tensor<2x4x3xf16>)
  // CHECK: hip.varlen_causal_conv_with_state(%[[CTX2]]) ins(%[[IN2]], %[[W2]], %[[CU2]], %[[ST2]] : tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, tensor<2x4x3xf16>) outs({{.*}}, {{.*}} : tensor<6x4xf16>, tensor<2x4x3xf16>)

  func.func @varlen_conv_dilated(
      %input: tensor<6x4xf16>,
      %weight: tensor<4x1x4xf16>,
      %cumulative: tensor<3xi32>,
      %state: tensor<2x4x6xf16>
  ) -> (tensor<6x4xf16>, tensor<2x4x6xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %final = "onnx.Custom"(%input, %weight, %cumulative, %none, %state) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      dilation = 2 : si64
    } : (tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, none, tensor<2x4x6xf16>)
      -> (tensor<6x4xf16>, tensor<2x4x6xf16>)
    return %output, %final : tensor<6x4xf16>, tensor<2x4x6xf16>
  }

  // CHECK-LABEL: func.func @varlen_conv_dilated
  // CHECK: hip.varlen_causal_conv_with_state
  // CHECK-SAME: {activation = "silu", dilation = 2 : i64}

  func.func @varlen_conv_state_capture(
      %input: tensor<6x4xf16>,
      %weight: tensor<4x1x4xf16>,
      %cumulative: tensor<3xi32>,
      %state: tensor<2x4x3xf16>,
      %capture: tensor<2xi32>
  ) -> (tensor<6x4xf16>, tensor<2x4x3xf16>, tensor<2x2x4xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %final, %update = "onnx.Custom"(%input, %weight, %cumulative, %none, %state, %capture) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft",
      state_update_capacity = 2 : si64
    } : (tensor<6x4xf16>, tensor<4x1x4xf16>, tensor<3xi32>, none, tensor<2x4x3xf16>, tensor<2xi32>)
      -> (tensor<6x4xf16>, tensor<2x4x3xf16>, tensor<2x2x4xf16>)
    return %output, %final, %update : tensor<6x4xf16>, tensor<2x4x3xf16>, tensor<2x2x4xf16>
  }

  // CHECK-LABEL: func.func @varlen_conv_state_capture
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.varlen_causal_conv_with_state
}
