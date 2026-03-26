// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX QMoE custom op is correctly lowered to hip.qmoe
// in tensor-first mode.
//
// This test validates:
// - QMoE lowering (onnx.Custom → hip.qmoe)
// - Required operands: input, router_probs, fc1_weights, fc1_scales,
//   fc2_weights, fc2_scales
// - Optional bias operands (fc1_bias, fc2_bias)
// - Attribute propagation (expert_weight_bits, k, block_size, activation, etc.)
// - Tensor-first DPS: tensor.empty() used as output init
// - Proper !hip.context threading through operations
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: QMoE with fc1_bias and fc2_bias =====

  func.func @main_graph(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc1_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<32x5760xf16>} : () -> tensor<32x5760xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    %fc2_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<32x2880xf16>} : () -> tensor<32x2880xf16>
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %fc1_b, %fc2_w, %fc2_s, %fc2_b) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
      activation_type = "swiglu",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_0"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, tensor<32x5760xf16>,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, tensor<32x2880xf16>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x2880xf16>, %[[ROUTER:.*]]: tensor<128x32xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x2880xf16>
  // CHECK: hip.qmoe(%[[CTX]]) ins(
  // CHECK-SAME: tensor<1x128x2880xf16>, tensor<128x32xf16>,
  // CHECK-SAME: tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>,
  // CHECK-SAME: tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>)
  // CHECK-SAME: fc1_bias(%{{.*}} : tensor<32x5760xf16>)
  // CHECK-SAME: fc2_bias(%{{.*}} : tensor<32x2880xf16>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x128x2880xf16>)
  // CHECK-SAME: activation_type = "swiglu"
  // CHECK-SAME: expert_weight_bits = 4
  // CHECK-SAME: k = 4
  // CHECK-NOT: onnx.Custom
}
