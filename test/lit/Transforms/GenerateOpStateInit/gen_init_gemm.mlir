// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.gemm. Its GemmState holds a shared_ptr to the device-wide hipBLASLt algo
// table (shared across sessions via WeakStore). The constructor takes no
// compile-time args (the algo cache fills lazily per GEMM shape). The generated
// init must allocate the slot array, call hipdnn_ep_op_state_construct_gemm,
// and store the result into slot 0. See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// CHECK: %[[RC:.*]] = llvm.select
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_gemm(%[[STATE]])
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hipdnn_ep_op_state_set(%[[STATE]], %[[SLOT]], %[[ST]])
// CHECK: llvm.return %[[RC]]

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
