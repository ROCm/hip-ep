// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.gemm. Its GemmState holds a shared_ptr to the device-wide hipBLASLt algo
// table (shared across sessions via WeakStore). The constructor takes no
// compile-time args (the algo cache fills lazily per GEMM shape). The generated
// init must allocate the slot array, then call hipdnn_ep_op_state_construct_gemm
// with slot 0, which stores the state into op_states[0] itself and returns an
// i8 ok flag. See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// On alloc failure, branch to the fail block and return without constructing.
// CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]], ^[[OK_BB:bb[0-9]+]]
// CHECK: ^[[OK_BB]]:
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_gemm(%[[STATE]], %[[SLOT]])
// CHECK: llvm.return
// CHECK: ^[[FAIL]]:
// CHECK: llvm.return

module {
  func.func @one_gemm(%ctx: !hip.context,
      %a: memref<128x256xf32, 1>,
      %b: memref<256x512xf32, 1>,
      %y: memref<128x512xf32, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<128x256xf32, 1>, memref<256x512xf32, 1>)
                   outs(%y : memref<128x512xf32, 1>)
    return
  }
}
