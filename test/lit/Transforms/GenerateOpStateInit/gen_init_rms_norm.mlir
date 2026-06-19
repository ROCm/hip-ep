// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.rms_norm. Its T5NormState holds a shared_ptr to the device-wide MIOpen
// descriptor table (shared across sessions via WeakStore). The constructor
// takes no compile-time args. The generated init must allocate the slot array,
// then call hipdnn_ep_op_state_construct_t5norm with slot 0, which stores the
// state into op_states[0] itself and returns an i8 ok flag. See
// docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// On alloc failure, branch to the fail block and return without constructing.
// CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]], ^[[OK_BB:bb[0-9]+]]
// CHECK: ^[[OK_BB]]:
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_t5norm(%[[STATE]], %[[SLOT]])
// CHECK: llvm.return
// CHECK: ^[[FAIL]]:
// CHECK: llvm.return

module {
  func.func @one_rms_norm(%ctx: !hip.context,
      %input: memref<1x128x4096xf16, 1>,
      %scale: memref<4096xf16, 1>,
      %output: memref<1x128x4096xf16, 1>) {
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<1x128x4096xf16, 1>, memref<4096xf16, 1>)
        outs(%output : memref<1x128x4096xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}
    return
  }
}
