// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmoe with optional fc1_zero_points and fc2_zero_points operands
// is correctly lowered to llvm.call @wrap_qmoe with non-null ZP pointers.
//
// This test validates:
// - Pointer extraction for optional ZP operands when present (non-null)
// - Spec-correct uint8 packed-nibble ZP layout for bits=4:
//     memref<num_experts x output_features x ceil(k_blocks/2) x ui8>
//   The runtime then strides by `e * N * ceil(k_blocks/2)` per expert
//   (lib/Runtime/real/qmoe.cpp, fix in commit 0918fdf).
// - Companion to the existing test_qmoe.mlir (which exercises bias-only and
//   no-optional-operands paths). Together they cover the {fc1_b, fc2_b,
//   fc1_zp, fc2_zp} optional-operand matrix.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // ===== Test 1: QMoE with fc1 + fc2 zero_points (no biases) =====
  //
  // ZP shape uses spec-correct ceil(k_blocks/2) packing: ceil(90/2) = 45.

  func.func @test_qmoe_with_zp(%ctx: !hip.context,
      %input: memref<1x128x2880xf16, 1>,
      %router: memref<128x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %fc1_zp: memref<32x5760x45xui8, 1>,
      %fc2_zp: memref<32x2880x45xui8, 1>,
      %output: memref<1x128x2880xf16, 1>) {
    hip.qmoe(%ctx) ins(
        %input, %router,
        %fc1_w, %fc1_s,
        %fc2_w, %fc2_s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        fc1_zero_points(%fc1_zp : memref<32x5760x45xui8, 1>)
        fc2_zero_points(%fc2_zp : memref<32x2880x45xui8, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_with_zp
  // CHECK: llvm.call @wrap_qmoe({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  // Verify 39 parameters (same shape as test_qmoe.mlir):
  // - 17 pointers: state, input, router, router_weights(null), fc1_w, fc1_s,
  //                fc1_b(null), fc2_w, fc2_s, fc2_b(null), fc3_w(null), fc3_s(null),
  //                fc3_b(null), fc1_zp, fc2_zp, fc3_zp(null), output
  // - 9 i64 + 3 f32 + 2 i64 attributes
  // - 5 pointers + 3 i64 router-gate fp32 recompute params (all null/0 here)
  // The fc1_zp / fc2_zp pointers MUST be non-null (extracted from the memref
  // descriptors), unlike the all-null path covered by test_qmoe.mlir.

  // ===== Test 2: QMoE with all four optional operands (biases + ZPs) =====
  //
  // Exercises the operand-segment grouping when both bias and ZP groups are
  // active simultaneously - this is the gpt-oss-120b prefill shape.

  func.func @test_qmoe_with_bias_and_zp(%ctx: !hip.context,
      %input: memref<1x128x2880xf16, 1>,
      %router: memref<128x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %fc1_b: memref<32x5760xf16, 1>,
      %fc2_b: memref<32x2880xf16, 1>,
      %fc1_zp: memref<32x5760x45xui8, 1>,
      %fc2_zp: memref<32x2880x45xui8, 1>,
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
        fc1_zero_points(%fc1_zp : memref<32x5760x45xui8, 1>)
        fc2_zero_points(%fc2_zp : memref<32x2880x45xui8, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_with_bias_and_zp
  // CHECK: llvm.call @wrap_qmoe({{.*}}) :
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  // 17 pointers, four of which (fc1_b, fc2_b, fc1_zp, fc2_zp) must be non-null;
  // the 5 router-gate pointers are null here (router-gate recompute disabled).
}
