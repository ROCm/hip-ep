// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.matmul_nbits. Its MatmulNbitsState owns a zero_points unpack cache that
// fills lazily per call, so the constructor takes no compile-time args. The
// generated init must (1) allocate the slot array and (2) call
// hipdnn_ep_op_state_construct_matmul_nbits with slot 0, which stores the state
// into op_states[0] itself and returns an i8 ok flag. See
// docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// The generated init function: (RuntimeState*) -> i32.
// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32

// Allocate the slot array for the single matmul_nbits (N = 1). _alloc returns a
// bool (i8); on failure we branch to the fail block and return without
// constructing anything (so no constructed state can leak).
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]], ^[[OK_BB:bb[0-9]+]]

// Success path: MatMulNBits contributes a no-arg construction (cache fills
// lazily at runtime). The construct fn takes slot 0 and stores the state into
// op_states[0] itself.
// CHECK: ^[[OK_BB]]:
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_matmul_nbits(%[[STATE]], %[[SLOT]])
// CHECK: llvm.return

// Failure path: nothing constructed, just return.
// CHECK: ^[[FAIL]]:
// CHECK: llvm.return

module {
  func.func @one_matmul_nbits(%ctx: !hip.context,
      %A: memref<1x128x2880xf16, 1>,
      %B: memref<5120x90x16xui8, 1>,
      %scales: memref<5120x90xf16, 1>,
      %output: memref<1x128x5120xf16, 1>) {
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<1x128x2880xf16, 1>, memref<5120x90x16xui8, 1>,
        memref<5120x90xf16, 1>)
        outs(%output : memref<1x128x5120xf16, 1>)
        {K = 2880 : i64, N = 5120 : i64, bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 0 : i64}
    return
  }
}
