// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.qmoe's optional router_weights operand reaches wrap_qmoe as a
// real pointer in argument slot 4 (0-based: state, input, router_probs,
// router_weights), rather than the null pointer emitted when it is absent.
//
// Companion: test_qmoe.mlir covers the absent case, where slot 4 is null.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_qmoe_router_weights(%ctx: !hip.context,
      %input: memref<1x128x2880xf16, 1>,
      %router: memref<128x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %router_w: memref<128x32xf16, 1>,
      %output: memref<1x128x2880xf16, 1>) {
    hip.qmoe(%ctx) ins(
        %input, %router,
        %fc1_w, %fc1_s,
        %fc2_w, %fc2_s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        router_weights(%router_w : memref<128x32xf16, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }

  // CHECK-LABEL: llvm.func @test_qmoe_router_weights

  // Only router_probs and router_weights are 2-D here, and the lowering
  // extracts router_probs first, so the second 2-D extract is router_weights.
  // CHECK: llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[RW_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[RW:.*]] = llvm.addrspacecast %[[RW_PTR]] : !llvm.ptr<1> to !llvm.ptr

  // Slot 4 (state, input, router_probs, router_weights) carries that pointer
  // instead of the llvm.mlir.zero used when router_weights is absent.
  // CHECK: llvm.call @wrap_qmoe(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[RW]],
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64) -> i32
}
