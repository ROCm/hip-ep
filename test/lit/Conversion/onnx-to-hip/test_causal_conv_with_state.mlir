// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft.CausalConvWithState is correctly lowered to
// hip.causal_conv_with_state in tensor-first mode.
//
// Test cases:
// 1. causal_conv_basic                   — 1D with bias and past_state, activation=silu
// 2. causal_conv_no_optional             — no bias, no past_state
// 3. causal_conv_bias_only               — with bias, no past_state
// 4. causal_conv_no_activation           — different shape, activation=none
// 5. causal_conv_dynamic                 — dynamic batch + seq_len (LLM decode)
// 6. causal_conv_dynamic_no_past_state   — dynamic input with no past_state
// 7. causal_conv_channels_last_dilated   — channels_last=1, dilation=3
// 8. causal_conv_state_window            — rejected
// 9. causal_conv_dilated_channels_first  — rejected
// 10. causal_conv_bf16                   — rejected
// 11. causal_conv_channels_last_bad_state — rejected
//
// All static cases assert:
// - context argument prepended
// - tensor.empty() for output init and present_state init
// - attributes forwarded (activation, ndim)
//
// Dynamic cases additionally assert:
// - tensor.dim extracts each dynamic dim at runtime
// - tensor.empty(%dim...) takes those sizes as dynamic operands
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
  // CHECK: hip.causal_conv_with_state(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]], %[[PS]] : tensor<1x64x128xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<1x64x3xf16>) outs({{.*}}, {{.*}} : tensor<1x64x128xf16>, tensor<1x64x3xf16>) {activation = "silu"}
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
  // 4. Different shape, activation=none
  // --------------------------------------------------------------------------
  func.func @causal_conv_no_activation(
      %input: tensor<2x128x64xf16>,
      %weight: tensor<128x1x3xf16>,
      %bias: tensor<128xf16>,
      %past_state: tensor<2x128x2xf16>
  ) -> (tensor<2x128x64xf16>, tensor<2x128x2xf16>) {
    %output, %present_state = "onnx.Custom"(%input, %weight, %bias, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      activation = "none",
      ndim = 1 : si64
    } : (tensor<2x128x64xf16>, tensor<128x1x3xf16>, tensor<128xf16>, tensor<2x128x2xf16>)
      -> (tensor<2x128x64xf16>, tensor<2x128x2xf16>)
    return %output, %present_state : tensor<2x128x64xf16>, tensor<2x128x2xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_no_activation
  // CHECK-SAME: !hip.context
  // CHECK: hip.causal_conv_with_state({{.*}}) ins({{.*}}) outs({{.*}})
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 5. Dynamic shape: batch + seq_len dynamic (typical LLM inference).
  //    Weight and kernel dim are static (k-1=3), past_state shares the
  //    dynamic batch axis.
  //
  //    Expectations:
  //    - output init: tensor.dim for dims 0 and 2 of input + tensor.empty with
  //      two dynamic sizes.
  //    - present_state init: tensor.dim for dim 0 of past_state + tensor.empty
  //      with one dynamic size.
  // --------------------------------------------------------------------------
  func.func @causal_conv_dynamic(
      %input: tensor<?x64x?xf16>,
      %weight: tensor<64x1x4xf16>,
      %bias: tensor<64xf16>,
      %past_state: tensor<?x64x3xf16>
  ) -> (tensor<?x64x?xf16>, tensor<?x64x3xf16>) {
    %output, %present_state = "onnx.Custom"(%input, %weight, %bias, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      ndim = 1 : si64
    } : (tensor<?x64x?xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<?x64x3xf16>)
      -> (tensor<?x64x?xf16>, tensor<?x64x3xf16>)
    return %output, %present_state : tensor<?x64x?xf16>, tensor<?x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_dynamic
  // CHECK-SAME: (%[[CTX5:.*]]: !hip.context, %[[IN5:.*]]: tensor<?x64x?xf16>, %[[W5:.*]]: tensor<64x1x4xf16>, %[[B5:.*]]: tensor<64xf16>, %[[PS5:.*]]: tensor<?x64x3xf16>)
  // output init: dims 0 and 2 of input extracted at runtime, fed into empty.
  // CHECK: tensor.dim %[[IN5]]
  // CHECK: tensor.dim %[[IN5]]
  // CHECK: tensor.empty({{.*}}) : tensor<?x64x?xf16>
  // present_state init: dim 0 of past_state extracted at runtime.
  // CHECK: tensor.dim %[[PS5]]
  // CHECK: tensor.empty({{.*}}) : tensor<?x64x3xf16>
  // CHECK: hip.causal_conv_with_state(%[[CTX5]]) ins(%[[IN5]], %[[W5]], %[[B5]], %[[PS5]] : tensor<?x64x?xf16>, tensor<64x1x4xf16>, tensor<64xf16>, tensor<?x64x3xf16>) outs({{.*}}, {{.*}} : tensor<?x64x?xf16>, tensor<?x64x3xf16>) {activation = "silu"
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 6. Fully dynamic without past_state: pastState is absent so present_state
  //    init derives dynamic sizes from input (batch). kernel_size (k-1=3)
  //    remains static because weight is static.
  // --------------------------------------------------------------------------
  func.func @causal_conv_dynamic_no_past_state(
      %input: tensor<?x64x?xf16>,
      %weight: tensor<64x1x4xf16>
  ) -> (tensor<?x64x?xf16>, tensor<?x64x3xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %none) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      ndim = 1 : si64
    } : (tensor<?x64x?xf16>, tensor<64x1x4xf16>, none, none)
      -> (tensor<?x64x?xf16>, tensor<?x64x3xf16>)
    return %output, %present_state : tensor<?x64x?xf16>, tensor<?x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_dynamic_no_past_state
  // CHECK-SAME: (%[[CTX6:.*]]: !hip.context, %[[IN6:.*]]: tensor<?x64x?xf16>, %[[W6:.*]]: tensor<64x1x4xf16>)
  // CHECK: tensor.dim %[[IN6]]
  // CHECK: tensor.empty({{.*}}) : tensor<?x64x?xf16>
  // present_state falls back to input for dim extraction (batch only).
  // CHECK: tensor.dim %[[IN6]]
  // CHECK: tensor.empty({{.*}}) : tensor<?x64x3xf16>
  // CHECK: hip.causal_conv_with_state(%[[CTX6]]) ins(%[[IN6]], %[[W6]] : tensor<?x64x?xf16>, tensor<64x1x4xf16>) outs({{.*}}, {{.*}} : tensor<?x64x?xf16>, tensor<?x64x3xf16>)
  // CHECK-NOT: hip.alloc

  func.func @causal_conv_channels_last_dilated(
      %input: tensor<?x?x64xf16>,
      %weight: tensor<64x1x4xf16>,
      %past_state: tensor<?x9x64xf16>
  ) -> (tensor<?x?x64xf16>, tensor<?x9x64xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      activation = "silu",
      channels_last = 1 : si64,
      dilation = 3 : si64,
      ndim = 1 : si64,
      state_window = 0 : si64
    } : (tensor<?x?x64xf16>, tensor<64x1x4xf16>, none, tensor<?x9x64xf16>)
      -> (tensor<?x?x64xf16>, tensor<?x9x64xf16>)
    return %output, %present_state : tensor<?x?x64xf16>, tensor<?x9x64xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_channels_last_dilated
  // CHECK: hip.causal_conv_with_state({{.*}}) ins({{.*}} : tensor<?x?x64xf16>, tensor<64x1x4xf16>, tensor<?x9x64xf16>) outs({{.*}} : tensor<?x?x64xf16>, tensor<?x9x64xf16>) {activation = "silu", channels_last = true, dilation = 3 : i64, state_channels_last = true}

  func.func @causal_conv_state_window(
      %input: tensor<1x64x8xf16>,
      %weight: tensor<64x1x4xf16>,
      %past_state: tensor<2x1x64x3xf16>
  ) -> (tensor<1x64x8xf16>, tensor<2x1x64x3xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      ndim = 1 : si64,
      state_window = 2 : si64
    } : (tensor<1x64x8xf16>, tensor<64x1x4xf16>, none, tensor<2x1x64x3xf16>)
      -> (tensor<1x64x8xf16>, tensor<2x1x64x3xf16>)
    return %output, %present_state : tensor<1x64x8xf16>, tensor<2x1x64x3xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_state_window
  // CHECK-NOT: hip.causal_conv_with_state
  // CHECK: onnx.Custom

  func.func @causal_conv_dilated_channels_first(
      %input: tensor<1x64x8xf16>,
      %weight: tensor<64x1x4xf16>,
      %past_state: tensor<1x64x6xf16>
  ) -> (tensor<1x64x8xf16>, tensor<1x64x6xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      dilation = 2 : si64,
      ndim = 1 : si64
    } : (tensor<1x64x8xf16>, tensor<64x1x4xf16>, none, tensor<1x64x6xf16>)
      -> (tensor<1x64x8xf16>, tensor<1x64x6xf16>)
    return %output, %present_state : tensor<1x64x8xf16>, tensor<1x64x6xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_dilated_channels_first
  // CHECK-NOT: hip.causal_conv_with_state
  // CHECK: onnx.Custom

  func.func @causal_conv_bf16(
      %input: tensor<1x64x8xbf16>,
      %weight: tensor<64x1x4xbf16>
  ) -> (tensor<1x64x8xbf16>, tensor<1x64x3xbf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %none) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      ndim = 1 : si64
    } : (tensor<1x64x8xbf16>, tensor<64x1x4xbf16>, none, none)
      -> (tensor<1x64x8xbf16>, tensor<1x64x3xbf16>)
    return %output, %present_state : tensor<1x64x8xbf16>, tensor<1x64x3xbf16>
  }

  // CHECK-LABEL: func.func @causal_conv_bf16
  // CHECK-NOT: hip.causal_conv_with_state
  // CHECK: onnx.Custom

  func.func @causal_conv_channels_last_bad_state(
      %input: tensor<1x8x64xf16>,
      %weight: tensor<64x1x4xf16>,
      %past_state: tensor<1x3x64xf16>
  ) -> (tensor<1x8x64xf16>, tensor<1x3x64xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %output, %present_state = "onnx.Custom"(%input, %weight, %none, %past_state) {
      function_name = "CausalConvWithState",
      domain_name = "com.microsoft",
      channels_last = 1 : si64,
      dilation = 2 : si64,
      ndim = 1 : si64
    } : (tensor<1x8x64xf16>, tensor<64x1x4xf16>, none, tensor<1x3x64xf16>)
      -> (tensor<1x8x64xf16>, tensor<1x3x64xf16>)
    return %output, %present_state : tensor<1x8x64xf16>, tensor<1x3x64xf16>
  }

  // CHECK-LABEL: func.func @causal_conv_channels_last_bad_state
  // CHECK-NOT: hip.causal_conv_with_state
  // CHECK: onnx.Custom
}
