// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.amd::QMoE nodes asking for a mode the runtime does not implement
// are NOT offloaded. Only activation_type="relu2" and routing_type="sigmoid"
// are implemented end to end; any other value must leave the onnx.Custom op in
// place so ORT keeps the node on CPU, rather than producing hip.qmoe_amd and
// silently computing relu2/sigmoid regardless of what the model asked for.
//
// The conversion runs as a greedy rewrite, so a declined match leaves the op
// untouched without failing the pass.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file %s | FileCheck %s

// Unsupported activation_type: everything else matches the supported case.
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
      activation_type = "swiglu", routing_type = "sigmoid",
      onnx_node_name = "QMoE_amd_bad_activation"
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
}

// CHECK: onnx.Custom
// CHECK-SAME: activation_type = "swiglu"
// CHECK-NOT: hip.qmoe_amd

// -----

// Unsupported routing_type, supported activation_type.
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
      activation_type = "relu2", routing_type = "softmax",
      onnx_node_name = "QMoE_amd_bad_routing"
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
}

// CHECK: onnx.Custom
// CHECK-SAME: routing_type = "softmax"
// CHECK-NOT: hip.qmoe_amd
