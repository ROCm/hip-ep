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
// - swiglu_fusion=0 converts only with activation_type="relu2"
// - weights_prepacked absent (-1) or 0 converts; 1 stays onnx.Custom
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

  // ===== Test 2: unfused relu2 experts with router_weights =====

  func.func @test_relu2_unfused(
      %input: tensor<1x4x2688xf16>,
      %router_probs: tensor<4x128xf16>,
      %router_weights: tensor<4x128xf16>) -> tensor<1x4x2688xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<128x1856x1344xui8>} : () -> tensor<128x1856x1344xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<128x1856x84xf16>} : () -> tensor<128x1856x84xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<128x2688x928xui8>} : () -> tensor<128x2688x928xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<128x2688x58xf16>} : () -> tensor<128x2688x58xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %router_weights) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
      activation_type = "relu2",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 6 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 0 : si64,
      use_sparse_mixer = 0 : si64, weights_prepacked = 0 : si64,
      onnx_node_name = "QMoE_relu2"
    } : (tensor<1x4x2688xf16>, tensor<4x128xf16>,
         tensor<128x1856x1344xui8>, tensor<128x1856x84xf16>, none,
         tensor<128x2688x928xui8>, tensor<128x2688x58xf16>,
         none, none, none, none, none, none, none, tensor<4x128xf16>) -> tensor<1x4x2688xf16>
    return %Y : tensor<1x4x2688xf16>
  }

  // CHECK-LABEL: func.func @test_relu2_unfused
  // CHECK: hip.qmoe
  // CHECK-SAME: tensor<128x1856x1344xui8>, tensor<128x1856x84xf16>,
  // CHECK-SAME: tensor<128x2688x928xui8>, tensor<128x2688x58xf16>)
  // CHECK-SAME: router_weights(%{{.*}} : tensor<4x128xf16>)
  // CHECK-SAME: activation_type = "relu2"
  // CHECK-SAME: k = 6
  // CHECK-SAME: swiglu_fusion = 0
  // CHECK-NOT: onnx.Custom

  // ===== Test 3: an unfused activation other than relu2 is not converted =====

  func.func @test_unfused_silu_rejected(
      %input: tensor<1x4x64xf16>,
      %router_probs: tensor<4x8xf16>) -> tensor<1x4x64xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<8x128x32xui8>} : () -> tensor<8x128x32xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8x128x2xf16>} : () -> tensor<8x128x2xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<8x64x64xui8>} : () -> tensor<8x64x64xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8x64x4xf16>} : () -> tensor<8x64x4xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_type = "silu",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 2 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 0 : si64,
      use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_silu"
    } : (tensor<1x4x64xf16>, tensor<4x8xf16>,
         tensor<8x128x32xui8>, tensor<8x128x2xf16>, none,
         tensor<8x64x64xui8>, tensor<8x64x4xf16>) -> tensor<1x4x64xf16>
    return %Y : tensor<1x4x64xf16>
  }

  // CHECK-LABEL: func.func @test_unfused_silu_rejected
  // CHECK-NOT: hip.qmoe
  // CHECK: onnx.Custom

  // ===== Test 4: weights_prepacked = 1 (an alternate prepacked layout) =====

  func.func @test_weights_prepacked_alternate_rejected(
      %input: tensor<1x4x64xf16>,
      %router_probs: tensor<4x8xf16>) -> tensor<1x4x64xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<8x128x32xui8>} : () -> tensor<8x128x32xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8x128x2xf16>} : () -> tensor<8x128x2xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<8x64x32xui8>} : () -> tensor<8x64x32xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8x64x2xf16>} : () -> tensor<8x64x2xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_type = "swiglu",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 2 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      use_sparse_mixer = 0 : si64, weights_prepacked = 1 : si64,
      onnx_node_name = "QMoE_prepacked_alternate"
    } : (tensor<1x4x64xf16>, tensor<4x8xf16>,
         tensor<8x128x32xui8>, tensor<8x128x2xf16>, none,
         tensor<8x64x32xui8>, tensor<8x64x2xf16>) -> tensor<1x4x64xf16>
    return %Y : tensor<1x4x64xf16>
  }

  // CHECK-LABEL: func.func @test_weights_prepacked_alternate_rejected
  // CHECK-NOT: hip.qmoe
  // CHECK: onnx.Custom
}
