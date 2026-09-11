// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the optional com.microsoft.QMoE router_weights input (operand index
// 14) is lowered to hip.qmoe's router_weights operand group.
//
// router_weights decouples selection from aggregation: router_probs still
// drives Top-K selection, while the mixing weights are gathered from
// router_weights at the selected expert indices. Both tensors therefore have
// the same [num_tokens, num_experts] shape, so this test also pins that the
// conversion reads index 14 rather than aliasing router_probs.
//
// The fc3_* slots (8, 9, 10) and the zero_point slots (11, 12, 13) are
// bypassed with onnx.NoValue placeholders so router_weights lands at 14.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>,
      %router_weights: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    // None placeholders for fc1_bias, fc2_bias, fc3_w, fc3_s, fc3_bias and the
    // three zero_point slots.
    %none_fc1b = "onnx.NoValue"() {value} : () -> none
    %none_fc2b = "onnx.NoValue"() {value} : () -> none
    %none_fc3w = "onnx.NoValue"() {value} : () -> none
    %none_fc3s = "onnx.NoValue"() {value} : () -> none
    %none_fc3b = "onnx.NoValue"() {value} : () -> none
    %none_fc1zp = "onnx.NoValue"() {value} : () -> none
    %none_fc2zp = "onnx.NoValue"() {value} : () -> none
    %none_fc3zp = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs,
                        %fc1_w, %fc1_s, %none_fc1b,
                        %fc2_w, %fc2_s, %none_fc2b,
                        %none_fc3w, %none_fc3s, %none_fc3b,
                        %none_fc1zp, %none_fc2zp, %none_fc3zp,
                        %router_weights) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
      activation_type = "swiglu",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_router_weights"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, none,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, none,
         none, none, none,
         none, none, none,
         tensor<128x32xf16>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x2880xf16>, %[[PROBS:.*]]: tensor<128x32xf16>, %[[WEIGHTS:.*]]: tensor<128x32xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x2880xf16>
  // CHECK: hip.qmoe(%[[CTX]]) ins(%[[INPUT]], %[[PROBS]]
  // CHECK-SAME: tensor<1x128x2880xf16>, tensor<128x32xf16>,
  // CHECK-SAME: tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>,
  // CHECK-SAME: tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>)
  // router_weights must reach its own operand group, distinct from the
  // router_probs value threaded into ins(...).
  // CHECK-SAME: router_weights(%[[WEIGHTS]] : tensor<128x32xf16>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x128x2880xf16>)
  // CHECK-NOT: onnx.Custom
}
