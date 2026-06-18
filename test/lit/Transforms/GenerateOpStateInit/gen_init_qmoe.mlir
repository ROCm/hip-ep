// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.qmoe. Unlike conv (which threads its compile-time kernel_shape), qmoe's
// QmoeState owns grow-on-demand device + pinned-host scratch sized lazily per
// call, so its constructor takes no compile-time args. The generated init must
// still (1) allocate the slot array, (2) call the no-arg constructor
// hipdnn_ep_op_state_construct_qmoe, and (3) store the result into slot 0.
// See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// The generated init function: (RuntimeState*) -> i32.
// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32

// Allocate the slot array for the single qmoe (N = 1). _alloc returns a bool
// (i8) that is mapped to the i32 status inference_init expects.
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: %[[RC:.*]] = llvm.select

// QMoE contributes a no-arg construction (buffers grow lazily at runtime).
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_qmoe(%[[STATE]])

// Store the constructed state into slot 0.
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hipdnn_ep_op_state_set(%[[STATE]], %[[SLOT]], %[[ST]])
// CHECK: llvm.return %[[RC]]

module {
  func.func @one_qmoe(%ctx: !hip.context,
      %input: memref<1x128x2880xf16, 1>,
      %router: memref<128x32xf16, 1>,
      %fc1_w: memref<32x5760x1440xui8, 1>,
      %fc1_s: memref<32x5760x90xf16, 1>,
      %fc2_w: memref<32x2880x1440xui8, 1>,
      %fc2_s: memref<32x2880x90xf16, 1>,
      %output: memref<1x128x2880xf16, 1>) {
    hip.qmoe(%ctx) ins(
        %input, %router,
        %fc1_w, %fc1_s,
        %fc2_w, %fc2_s :
        memref<1x128x2880xf16, 1>, memref<128x32xf16, 1>,
        memref<32x5760x1440xui8, 1>, memref<32x5760x90xf16, 1>,
        memref<32x2880x1440xui8, 1>, memref<32x2880x90xf16, 1>)
        outs(%output : memref<1x128x2880xf16, 1>)
        {expert_weight_bits = 4 : i64, k = 4 : i64, block_size = 32 : i64,
         normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
         use_sparse_mixer = 0 : i64,
         activation_alpha = 1.702000e+00 : f32, activation_beta = 1.000000e+00 : f32,
         swiglu_limit = 7.000000e+00 : f32, activation_type = "swiglu"}
    return
  }
}
