// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the QMoE conversion derives `block_size` from operand shapes when
// the attribute is missing on the ONNX custom op (regression coverage for
// commit 3c503f5 "QMoEConversion: derive block_size from fc1_scales when
// the attribute is absent").
//
// AWQ exports of gpt-oss-120b (and similar models) omit the `block_size`
// attribute from QMoE nodes; without this derivation `wrap_qmoe` later
// divides by zero. The conversion derives:
//
//     block_size = ceil(hidden_size / k_blocks)
//
// where `hidden_size = input.shape.back()` and
// `k_blocks = fc1_scales.shape.back()`.
//
// This test validates two shape configurations:
//   1. Per-group: fc1_scales [E, N, 90], hidden=2880 -> block_size = 32
//   2. Per-channel: fc1_scales [E, N, 1], hidden=2880 -> block_size = 2880
//      (succeeds in the conversion pass; runtime guard in
//      lib/Runtime/real/qmoe.cpp may further restrict it)
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: Per-group scales [E, N, 90], derive block_size = 32 =====
  //
  // Note: NO `block_size` attribute on the onnx.Custom op below. The
  // conversion must derive it from input.shape.back()=2880 divided by
  // fc1_scales.shape.back()=90, yielding 32.
  func.func @main_graph(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    // QMoE conversion requires >= 7 operands; insert a NoValue at the
    // fc1_bias slot (operand index 4) so we satisfy the count without
    // adding a real bias tensor.
    %none_fc1b = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none_fc1b, %fc2_w, %fc2_s) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
      activation_type = "swiglu",
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_no_block_size_per_group"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, none,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.qmoe
  // CHECK-SAME: block_size = 32
  // CHECK-NOT: onnx.Custom

  // ===== Test 2: Per-channel scales [E, N, 1], derive block_size = 2880 =====
  //
  // With one scale per output row (k_blocks=1), the derived block_size
  // collapses to hidden_size. Conversion still succeeds; this documents the
  // contract that the pass is shape-driven and does not impose its own
  // even/positive constraint on the derived value (the runtime does).
  func.func @test_per_channel(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x1xf16>} : () -> tensor<32x5760x1xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x1xf16>} : () -> tensor<32x2880x1xf16>
    %none_fc1b = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs, %fc1_w, %fc1_s, %none_fc1b, %fc2_w, %fc2_s) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
      activation_type = "swiglu",
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_no_block_size_per_channel"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x1xf16>, none,
         tensor<32x2880x1440xui8>, tensor<32x2880x1xf16>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @test_per_channel
  // CHECK: hip.qmoe
  // CHECK-SAME: block_size = 2880
  // CHECK-NOT: onnx.Custom
}
