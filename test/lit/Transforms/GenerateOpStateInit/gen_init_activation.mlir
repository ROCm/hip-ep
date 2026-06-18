// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.sigmoid. Its ActivationState holds a shared_ptr to the device-wide MIOpen
// descriptor table (shared across sessions via WeakStore, and shared by
// hip.sigmoid / hip.tanh / hip.softplus). The constructor takes no compile-time
// args. The generated init must (1) allocate the slot array, (2) call the
// no-arg constructor hipdnn_ep_op_state_construct_activation, and (3) store the
// result into slot 0. See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: %[[RC:.*]] = llvm.select
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_activation(%[[STATE]])
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hipdnn_ep_op_state_set(%[[STATE]], %[[SLOT]], %[[ST]])
// CHECK: llvm.return %[[RC]]

module {
  func.func @one_sigmoid(%ctx: !hip.context,
      %input: memref<1x128x512xf32, 1>,
      %output: memref<1x128x512xf32, 1>) {
    hip.sigmoid(%ctx) ins(%input : memref<1x128x512xf32, 1>)
                      outs(%output : memref<1x128x512xf32, 1>)
    return
  }
}
