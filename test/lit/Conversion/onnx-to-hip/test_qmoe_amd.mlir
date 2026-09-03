// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX com.amd::QMoE custom op is correctly lowered to hip.qmoe_amd.
//
// This test validates:
// - QMoE (com.amd) lowering (onnx.Custom -> hip.qmoe_amd), distinct from the
//   com.microsoft::QMoE -> hip.qmoe path (test_qmoe.mlir)
// - All 15 required operands are threaded through positionally (no optional
//   operands in this schema)
// - Attribute propagation (k, expert_weight_bits, block_size,
//   normalize_routing_weights, use_correction_bias, routed_scaling_factor,
//   activation_type, routing_type)
// - Tensor-first DPS: tensor.empty() used as output init
// - Proper !hip.context threading through operations
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(
      %hidden_states: tensor<1x4x16xf16>) -> tensor<1x4x16xf16> {
    %fc1e_w = "onnx.Constant"() {value = dense<1> : tensor<4x8x1x4xui8>} : () -> tensor<4x8x1x4xui8>
    %fc1e_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4x8x1xf16>} : () -> tensor<4x8x1xf16>
    %fc2e_w = "onnx.Constant"() {value = dense<1> : tensor<4x8x1x4xui8>} : () -> tensor<4x8x1x4xui8>
    %fc2e_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4x8x1xf16>} : () -> tensor<4x8x1xf16>
    %fc1l_w = "onnx.Constant"() {value = dense<1> : tensor<8x2x4xui8>} : () -> tensor<8x2x4xui8>
    %fc1l_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8x2xf16>} : () -> tensor<8x2xf16>
    %fc2l_w = "onnx.Constant"() {value = dense<1> : tensor<16x1x4xui8>} : () -> tensor<16x1x4xui8>
    %fc2l_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x1xf16>} : () -> tensor<16x1xf16>
    %sh1_w = "onnx.Constant"() {value = dense<1> : tensor<16x2x4xui8>} : () -> tensor<16x2x4xui8>
    %sh1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x2xf16>} : () -> tensor<16x2xf16>
    %sh2_w = "onnx.Constant"() {value = dense<1> : tensor<16x2x4xui8>} : () -> tensor<16x2x4xui8>
    %sh2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x2xf16>} : () -> tensor<16x2xf16>
    %router_w = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x4xf16>} : () -> tensor<16x4xf16>
    %corr_b = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<4xf16>} : () -> tensor<4xf16>
    %Y = "onnx.Custom"(%hidden_states, %fc1e_w, %fc1e_s, %fc2e_w, %fc2e_s,
                        %fc1l_w, %fc1l_s, %fc2l_w, %fc2l_s,
                        %sh1_w, %sh1_s, %sh2_w, %sh2_s,
                        %router_w, %corr_b) {
      function_name = "QMoE",
      domain_name = "com.amd",
      k = 2 : si64, expert_weight_bits = 4 : si64, block_size = 8 : si64,
      normalize_routing_weights = 1 : si64, use_correction_bias = 1 : si64,
      routed_scaling_factor = 5.000000e+00 : f32,
      activation_type = "relu2", routing_type = "sigmoid",
      onnx_node_name = "QMoE_amd_0"
    } : (tensor<1x4x16xf16>,
         tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
         tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
         tensor<8x2x4xui8>, tensor<8x2xf16>,
         tensor<16x1x4xui8>, tensor<16x1xf16>,
         tensor<16x2x4xui8>, tensor<16x2xf16>,
         tensor<16x2x4xui8>, tensor<16x2xf16>,
         tensor<16x4xf16>, tensor<4xf16>) -> tensor<1x4x16xf16>
    return %Y : tensor<1x4x16xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[HS:.*]]: tensor<1x4x16xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x4x16xf16>
  // CHECK: hip.qmoe_amd(%[[CTX]]) ins(
  // CHECK-SAME: tensor<1x4x16xf16>,
  // CHECK-SAME: tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
  // CHECK-SAME: tensor<4x8x1x4xui8>, tensor<4x8x1xf16>,
  // CHECK-SAME: tensor<8x2x4xui8>, tensor<8x2xf16>,
  // CHECK-SAME: tensor<16x1x4xui8>, tensor<16x1xf16>,
  // CHECK-SAME: tensor<16x2x4xui8>, tensor<16x2xf16>,
  // CHECK-SAME: tensor<16x2x4xui8>, tensor<16x2xf16>,
  // CHECK-SAME: tensor<16x4xf16>, tensor<4xf16>
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x4x16xf16>)
  // Attributes print in alphabetical order.
  // CHECK-SAME: activation_type = "relu2"
  // CHECK-SAME: block_size = 8
  // CHECK-SAME: expert_weight_bits = 4
  // CHECK-SAME: k = 2
  // CHECK-SAME: normalize_routing_weights = 1
  // CHECK-SAME: routed_scaling_factor = 5.000000e+00
  // CHECK-SAME: routing_type = "sigmoid"
  // CHECK-SAME: use_correction_bias = 1
  // CHECK-NOT: onnx.Custom
}
