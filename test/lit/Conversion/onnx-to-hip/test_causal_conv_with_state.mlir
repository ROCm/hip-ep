// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.CausalConvWithState is correctly lowered to
// hip.causal_conv_with_state in tensor-first mode.
//
// Test cases:
// 1. causal_conv_basic        — 1D with bias and past_state, activation=silu
// 2. causal_conv_no_optional  — no bias, no past_state
// 3. causal_conv_bias_only    — with bias, no past_state
// 4. causal_conv_f32          — f32 element type
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init and present_state init
// - attributes forwarded (activation, ndim)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x64x128xf16>) -> tensor<1x64x128xf16> {
    return %arg0 : tensor<1x64x128xf16>
  }

  // --------------------------------------------------------------------------
  // 1. Basic 1D causal conv with bias, past_state, activation=silu
  // --------------------------------------------------------------------------
  func.func @causal_conv_basic(
      %input: tensor<1x64x128xf16>,
      %weight: tensor<64x1x4xf16>,
      %bias: tensor<64xf16>,
      %past_state: tensor<1x64x3xf16>
  ) -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>) {
    %output, %present_state = "onnx.Custom"(%input, %weight, %bias, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      ndim = 1 : si64
    } : (tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<1x64x3xf16>)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>)
    return %output, %present_state : tensor<1x64x128xf16>, tensor<1x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x128xf16>, %[[W:.*]]: tensor<64x1x4xf16>, %[[B:.*]]: tensor<64xf16>, %[[PS:.*]]: tensor<1x64x3xf16>)
  // CHECK: tensor.empty() : tensor<1x64x128xf16>
  // CHECK: tensor.empty() : tensor<1x64x3xf16>
  // CHECK: hip.causal_conv_with_state(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]], %[[PS]] : tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<1x64x3xf16>) outs({{.*}}, {{.*}} : tensor<1x64x128xf16>, tensor<1x64x3xf16>) {activation = "silu", ndim = 1 : i64}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. No optional inputs (no bias, no past_state)
  // --------------------------------------------------------------------------
  func.func @causal_conv_no_optional(
      %input: tensor<1x64x128xf16>,
      %weight: tensor<64x1x4xf16>
  ) -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %none) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      ndim = 1 : si64
    } : (tensor<1x64x128xf16>, tensor<64x1x4xf16>, none, none)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>)
    return %output, %present_state : tensor<1x64x128xf16>, tensor<1x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_no_optional
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[IN2:.*]]: tensor<1x64x128xf16>, %[[W2:.*]]: tensor<64x1x4xf16>)
  // CHECK: hip.causal_conv_with_state(%[[CTX2]]) ins(%[[IN2]], %[[W2]] : tensor<1x64x128xf16>, tensor<64x1x4xf16>) outs({{.*}}, {{.*}} : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. With bias only (no past_state)
  // --------------------------------------------------------------------------
  func.func @causal_conv_bias_only(
      %input: tensor<1x64x128xf16>,
      %weight: tensor<64x1x4xf16>,
      %bias: tensor<64xf16>
  ) -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %bias, %none) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      ndim = 1 : si64
    } : (tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>, none)
      -> (tensor<1x64x128xf16>, tensor<1x64x3xf16>)
    return %output, %present_state : tensor<1x64x128xf16>, tensor<1x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_bias_only
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[IN3:.*]]: tensor<1x64x128xf16>, %[[W3:.*]]: tensor<64x1x4xf16>, %[[B3:.*]]: tensor<64xf16>)
  // CHECK: hip.causal_conv_with_state(%[[CTX3]]) ins(%[[IN3]], %[[W3]], %[[B3]] : tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>) outs({{.*}}, {{.*}} : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 4. f32 element type
  // --------------------------------------------------------------------------
  func.func @causal_conv_f32(
      %input: tensor<2x128x64xf32>,
      %weight: tensor<128x1x3xf32>,
      %bias: tensor<128xf32>,
      %past_state: tensor<2x128x2xf32>
  ) -> (tensor<2x128x64xf32>, tensor<2x128x2xf32>) {
    %output, %present_state = "onnx.Custom"(%input, %weight, %bias, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      activation = "none",
      ndim = 1 : si64
    } : (tensor<2x128x64xf32>, tensor<128x1x3xf32>, tensor<128xf32>, tensor<2x128x2xf32>)
      -> (tensor<2x128x64xf32>, tensor<2x128x2xf32>)
    return %output, %present_state : tensor<2x128x64xf32>, tensor<2x128x2xf32>
  }

  // CHECK-LABEL: func.func @causal_conv_f32
  // CHECK-SAME: !hip.context
  // CHECK: hip.causal_conv_with_state({{.*}}) ins({{.*}}) outs({{.*}}) {activation = "none", ndim = 1 : i64}
  // CHECK-NOT: hip.alloc
}
