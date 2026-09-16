// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// the elementwise ops (hip.add / hip.mul / hip.min / hip.max), which all
// lower to wrap_elementwise and share one (empty) OpTensorState type -- the
// slot exists only because every OpStateOpInterface op must construct one.
// The constructor takes no compile-time args. Here two instances (one add,
// one mul) each get their own slot, both built by
// hipdnn_ep_op_state_construct_optensor. See
// docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(2 : i64)
// CHECK: llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// On alloc failure, branch to the fail block and return without constructing.
// CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]], ^[[OK_BB:bb[0-9]+]]
// Each instance is constructed in its own slot (anchored inside the construct
// block so the slot constants are not confused with the entry-block i32
// success/failure return constants); the construct fn takes the slot operand
// and stores the state into op_states[slot] itself (no separate _set call).
// CHECK: ^[[OK_BB]]:
// CHECK: %[[SLOT0:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ST0:.*]] = llvm.call @hipdnn_ep_op_state_construct_optensor(%[[STATE]], %[[SLOT0]])
// CHECK: %[[SLOT1:.*]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[ST1:.*]] = llvm.call @hipdnn_ep_op_state_construct_optensor(%[[STATE]], %[[SLOT1]])

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
