// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX QMoE custom op with optional zero_points operands (asymmetric
// quantization) is correctly lowered to hip.qmoe in tensor-first mode.
//
// This test validates:
// - QMoE lowering with both fc1_zero_points and fc2_zero_points operands
//   present (operand indices 11 and 12 of the 14-input ONNX schema)
// - Spec-correct uint8 packed-nibble zero_points layout for bits=4:
//     shape [num_experts, output_features, ceil(k_blocks/2)]
//   (NOT the unpacked [num_experts, output_features, k_blocks] layout)
// - The fc3_* slots (operand indices 8, 9, 10) are correctly bypassed via
//   onnx.NoValue placeholders so the ZP operands land at the right indices
// - Both ZP operands are propagated through the conversion to the hip.qmoe
//   `fc1_zero_points` and `fc2_zero_points` operand groups
//
// Companion runtime: lib/Runtime/real/qmoe.cpp slices these per-expert with
// stride `e * N * ((k_blocks + 1) / 2)` matching the packed-nibble layout
// (commit 0918fdf "fix(qmoe): use packed-nibble stride for per-expert
// zero_points slice").
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== QMoE with fc1 and fc2 zero_points (no biases, no fc3) =====
  //
  // Shapes (matching gpt-oss-120b family but with num_experts=32 for brevity):
  //   input          : [1, 128, 2880]                       hidden_size = 2880
  //   router_probs   : [128, 32]                            num_experts = 32
  //   fc1 weights    : [32, 5760, 1440]  packed uint4       fusion_inter = 5760
  //                                                         k_blocks * blob = 90*16 = 1440
  //   fc1 scales     : [32, 5760, 90]    fp16, per-group    k_blocks = 90
  //   fc1 zero_points: [32, 5760, 45]    uint8 packed nibbles, ceil(90/2) = 45
  //   fc2 weights    : [32, 2880, 1440]
  //   fc2 scales     : [32, 2880, 90]
  //   fc2 zero_points: [32, 2880, 45]
  //
  // ONNX QMoE operand order (14 slots, missing optional ones use onnx.NoValue):
  //   [0]  input
  //   [1]  router_probs
  //   [2]  fc1_weights
  //   [3]  fc1_scales
  //   [4]  fc1_bias        (none here)
  //   [5]  fc2_weights
  //   [6]  fc2_scales
  //   [7]  fc2_bias        (none here)
  //   [8]  fc3_weights     (none - swiglu_fusion=1)
  //   [9]  fc3_scales      (none - swiglu_fusion=1)
  //   [10] fc3_bias        (none - swiglu_fusion=1)
  //   [11] fc1_zero_points (PRESENT)
  //   [12] fc2_zero_points (PRESENT)
  //   [13] fc3_zero_points (none - swiglu_fusion=1)
  func.func @main_graph(
      %input: tensor<1x128x2880xf16>,
      %router_probs: tensor<128x32xf16>) -> tensor<1x128x2880xf16> {
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<32x5760x1440xui8>} : () -> tensor<32x5760x1440xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x5760x90xf16>} : () -> tensor<32x5760x90xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<32x2880x1440xui8>} : () -> tensor<32x2880x1440xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x2880x90xf16>} : () -> tensor<32x2880x90xf16>
    // Spec-correct packed-nibble layout: ceil(k_blocks/2) = ceil(90/2) = 45
    %fc1_zp = "onnx.Constant"() {value = dense<8> : tensor<32x5760x45xui8>} : () -> tensor<32x5760x45xui8>
    %fc2_zp = "onnx.Constant"() {value = dense<8> : tensor<32x2880x45xui8>} : () -> tensor<32x2880x45xui8>
    // None placeholders for fc1_bias, fc2_bias, fc3_w, fc3_s, fc3_bias
    %none_fc1b = "onnx.NoValue"() {value} : () -> none
    %none_fc2b = "onnx.NoValue"() {value} : () -> none
    %none_fc3w = "onnx.NoValue"() {value} : () -> none
    %none_fc3s = "onnx.NoValue"() {value} : () -> none
    %none_fc3b = "onnx.NoValue"() {value} : () -> none
    %Y = "onnx.Custom"(%input, %router_probs,
                        %fc1_w, %fc1_s, %none_fc1b,
                        %fc2_w, %fc2_s, %none_fc2b,
                        %none_fc3w, %none_fc3s, %none_fc3b,
                        %fc1_zp, %fc2_zp) {
      function_name = "QMoE",
      domain_name = "com.microsoft",
      activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
      activation_type = "swiglu",
      block_size = 32 : si64, expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      swiglu_limit = 7.000000e+00 : f32, use_sparse_mixer = 0 : si64,
      onnx_node_name = "QMoE_packed_zp"
    } : (tensor<1x128x2880xf16>, tensor<128x32xf16>,
         tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>, none,
         tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>, none,
         none, none, none,
         tensor<32x5760x45xui8>, tensor<32x2880x45xui8>) -> tensor<1x128x2880xf16>
    return %Y : tensor<1x128x2880xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x2880xf16>, %[[ROUTER:.*]]: tensor<128x32xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x2880xf16>
  // CHECK: hip.qmoe(%[[CTX]]) ins(
  // CHECK-SAME: tensor<1x128x2880xf16>, tensor<128x32xf16>,
  // CHECK-SAME: tensor<32x5760x1440xui8>, tensor<32x5760x90xf16>,
  // CHECK-SAME: tensor<32x2880x1440xui8>, tensor<32x2880x90xf16>)
  // The packed-nibble zero_points must reach hip.qmoe's named operand groups
  // with their spec-correct shapes preserved.
  // CHECK-SAME: fc1_zero_points(%{{.*}} : tensor<32x5760x45xui8>)
  // CHECK-SAME: fc2_zero_points(%{{.*}} : tensor<32x2880x45xui8>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<1x128x2880xf16>)
  // CHECK-SAME: expert_weight_bits = 4
  // CHECK-SAME: k = 4
  // CHECK-NOT: onnx.Custom
}
