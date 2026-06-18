// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits the standalone op-state init function
// that (1) allocates the slot array, (2) lets each stateful op contribute its
// own parameterized construction via generateOpStateInit, and (3) stores the
// constructed pointer into its slot. The conv op is the reference op: it reads
// its compile-time kernel_shape [7, 7] and passes it as i64 args to the
// runtime constructor hipdnn_ep_op_state_construct_conv.
// See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// The generated init function: (RuntimeState*) -> i32.
// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32

// Allocate the slot array for the single conv (N = 1). _alloc returns a bool
// (i8) that is mapped to the i32 status inference_init expects (0 = success).
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: %[[RC:.*]] = llvm.select

// Conv contributes parameterized construction from its kernel_shape [7, 7].
// CHECK-DAG: %[[KH:.*]] = llvm.mlir.constant(7 : i64)
// CHECK-DAG: %[[KW:.*]] = llvm.mlir.constant(7 : i64)
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_conv(%[[STATE]], %{{.*}}, %{{.*}})

// Store the constructed state into slot 0.
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hipdnn_ep_op_state_set(%[[STATE]], %[[SLOT]], %[[ST]])
// CHECK: llvm.return %[[RC]]

module {
  func.func @one_conv(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}
    return
  }
}
