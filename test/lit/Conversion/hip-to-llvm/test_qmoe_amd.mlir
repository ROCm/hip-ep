// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmoe_amd is correctly lowered to an LLVM call to wrap_qmoe_amd
// runtime function. Distinct from hip.qmoe -> wrap_qmoe (test_qmoe.mlir):
// com.amd::QMoE has no optional operands and carries extra latent/shared
// weight operands plus derived latent_size/moe_intermediate_size/
// shared_intermediate_size dimensions.
//
// This test validates:
// - Runtime call generation (hip.qmoe_amd -> llvm.call @wrap_qmoe_amd)
// - Pointer extraction for all 15 required operands + output (17 pointers
//   total, including the state handle)
// - Dimension extraction (num_tokens, hidden_size, latent_size,
//   moe_intermediate_size, shared_intermediate_size, num_experts)
// - Attribute-to-constant lowering (k, expert_weight_bits, block_size,
//   normalize_routing_weights, use_correction_bias, routed_scaling_factor)
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_qmoe_amd(%ctx: !hip.context,
      %hidden_states: memref<1x4x16xf16, 1>,
      %fc1e_w: memref<4x8x1x4xui8, 1>,
      %fc1e_s: memref<4x8x1xf16, 1>,
      %fc2e_w: memref<4x8x1x4xui8, 1>,
      %fc2e_s: memref<4x8x1xf16, 1>,
      %fc1l_w: memref<8x2x4xui8, 1>,
      %fc1l_s: memref<8x2xf16, 1>,
      %fc2l_w: memref<16x1x4xui8, 1>,
      %fc2l_s: memref<16x1xf16, 1>,
      %sh1_w: memref<16x2x4xui8, 1>,
      %sh1_s: memref<16x2xf16, 1>,
      %sh2_w: memref<16x2x4xui8, 1>,
      %sh2_s: memref<16x2xf16, 1>,
      %router_w: memref<16x4xf16, 1>,
      %corr_b: memref<4xf16, 1>,
      %output: memref<1x4x16xf16, 1>) {
    hip.qmoe_amd(%ctx) ins(
        %hidden_states,
        %fc1e_w, %fc1e_s,
        %fc2e_w, %fc2e_s,
        %fc1l_w, %fc1l_s,
        %fc2l_w, %fc2l_s,
        %sh1_w, %sh1_s,
        %sh2_w, %sh2_s,
        %router_w, %corr_b :
        memref<1x4x16xf16, 1>,
        memref<4x8x1x4xui8, 1>, memref<4x8x1xf16, 1>,
        memref<4x8x1x4xui8, 1>, memref<4x8x1xf16, 1>,
        memref<8x2x4xui8, 1>, memref<8x2xf16, 1>,
        memref<16x1x4xui8, 1>, memref<16x1xf16, 1>,
        memref<16x2x4xui8, 1>, memref<16x2xf16, 1>,
        memref<16x2x4xui8, 1>, memref<16x2xf16, 1>,
        memref<16x4xf16, 1>, memref<4xf16, 1>)
        outs(%output : memref<1x4x16xf16, 1>)
        {k = 2 : i64, expert_weight_bits = 4 : i64, block_size = 8 : i64,
         normalize_routing_weights = 1 : i64, use_correction_bias = 1 : i64,
         routed_scaling_factor = 5.000000e+00 : f32,
         activation_type = "relu2", routing_type = "sigmoid"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_amd
  // CHECK: llvm.call @wrap_qmoe_amd({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
  // Verify 30 parameters:
  // - 17 pointers: state, hidden_states, fc1_experts_weights,
  //   fc1_experts_scales, fc2_experts_weights, fc2_experts_scales,
  //   fc1_latent_weights, fc1_latent_scales, fc2_latent_weights,
  //   fc2_latent_scales, shared_fc1_weights, shared_fc1_scales,
  //   shared_fc2_weights, shared_fc2_scales, router_weight, correction_bias,
  //   output
  // - 12 i64 + 1 f32: num_tokens, hidden_size, latent_size,
  //   moe_intermediate_size, shared_intermediate_size, num_experts, k,
  //   expert_weight_bits, block_size, normalize_routing_weights,
  //   use_correction_bias, routed_scaling_factor, elem_size
}
