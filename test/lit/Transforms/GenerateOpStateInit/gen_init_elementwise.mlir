// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// the MIOpen OpTensor elementwise ops (hip.add / hip.mul / hip.min / hip.max),
// which all lower to wrap_miopenOpTensor and share one OpTensorState (a
// shared_ptr to the device-wide MIOpen descriptor table, shared across sessions
// via WeakStore). The constructor takes no compile-time args. Here two
// instances (one add, one mul) each get their own slot, both built by
// hipdnn_ep_op_state_construct_optensor. See
// docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(2 : i64)
// CHECK: llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: llvm.call @hipdnn_ep_op_state_construct_optensor(%[[STATE]])
// CHECK: llvm.call @hipdnn_ep_op_state_construct_optensor(%[[STATE]])

module {
  func.func @add_then_mul(%ctx: !hip.context,
      %a: memref<128x512xf32, 1>,
      %b: memref<128x512xf32, 1>,
      %c: memref<128x512xf32, 1>,
      %d: memref<128x512xf32, 1>) {
    hip.add(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%c : memref<128x512xf32, 1>)
    hip.mul(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%d : memref<128x512xf32, 1>)
    return
  }
}
