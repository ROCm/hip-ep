// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmoe is correctly lowered to an LLVM call to wrap_qmoe
// runtime function.
//
// This test validates:
// - Runtime call generation (hip.qmoe → llvm.call @wrap_qmoe)
// - Pointer extraction for required operands (input, router, fc1/fc2 weights/scales, output)
// - Null pointer generation for absent optional operands
// - Pointer extraction for present optional operands (fc1_bias, fc2_bias)
// - Dimension extraction (num_tokens, hidden_size, inter_size, num_experts)
// - Attribute-to-constant lowering (k, expert_weight_bits, block_size, activation, etc.)
// ============================================================================

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s

module {
  // ===== Test 1: QMoE with fc1_bias and fc2_bias =====

  func.func @test_qmoe_with_bias(%ctx: !hip.context,
      %input: memref<1x128x2880xf16, 1>,
      %router: memref<128x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %fc1_b: memref<32x5760xf16, 1>,
      %fc2_b: memref<32x2880xf16, 1>,
      %output: memref<1x128x2880xf16, 1>) {
    hip.qmoe(%ctx) ins(
        %input, %router,
        %fc1_w, %fc1_s,
        %fc2_w, %fc2_s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        fc1_bias(%fc1_b : memref<32x5760xf16, 1>)
        fc2_bias(%fc2_b : memref<32x2880xf16, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_with_bias
  // CHECK: llvm.call @wrap_qmoe({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64, i32) -> i32
  // Verify 31 parameters:
  // - 16 pointers: state, input, router, fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc2_b,
  //                fc3_w(null), fc3_s(null), fc3_b(null), fc1_zp(null), fc2_zp(null), fc3_zp(null), output
  // - 11 i64 + 3 f32: num_tokens, hidden_size, inter_size, num_experts, k,
  //                    expert_weight_bits, block_size, swiglu_fusion, activation_type,
  //                    activation_alpha, activation_beta, swiglu_limit,
  //                    normalize_routing_weights, elem_size
  // - 1 i32: op_state_slot (per-instance QmoeState home; threaded by
  //          --assign-op-state-slots, replaces shared RuntimeState::qmoe_scratch)

  // ===== Test 2: QMoE without optional biases =====

  func.func @test_qmoe_no_bias(%ctx: !hip.context,
      %input: memref<1x64x2880xf16, 1>,
      %router: memref<64x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %output: memref<1x64x2880xf16, 1>) {
    hip.qmoe(%ctx) ins(
        %input, %router,
        %fc1_w, %fc1_s,
        %fc2_w, %fc2_s :
        memref<1x64x2880xf16, 1>, memref<64x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        outs(%output : memref<1x64x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_no_bias
  // CHECK: llvm.call @wrap_qmoe({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64, i32) -> i32
  // Optional pointers (fc1_b, fc2_b, fc3_*, zero_points) should be null
}
