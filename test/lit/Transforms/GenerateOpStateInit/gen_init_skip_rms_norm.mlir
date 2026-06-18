// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --generate-op-state-init emits per-instance op-state construction for
// hip.skip_rms_norm. Its SkipT5NormState holds a shared_ptr to the device-wide
// MIOpen descriptor table (shared across sessions via WeakStore). The
// constructor takes no compile-time args. The generated init must allocate the
// slot array, call hipdnn_ep_op_state_construct_skip_t5norm, and store the
// result into slot 0. See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --generate-op-state-init | FileCheck %s

// CHECK: llvm.func @hipdnn_ep_op_states_init_fn(%[[STATE:.*]]: !llvm.ptr) -> i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: %[[OK:.*]] = llvm.call @hipdnn_ep_op_states_alloc(%[[STATE]], %[[N]])
// On alloc failure, branch to the fail block and return without constructing.
// CHECK: llvm.cond_br %{{.*}}, ^[[FAIL:bb[0-9]+]], ^[[OK_BB:bb[0-9]+]]
// CHECK: ^[[OK_BB]]:
// CHECK: %[[ST:.*]] = llvm.call @hipdnn_ep_op_state_construct_skip_t5norm(%[[STATE]])
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32)
// CHECK: llvm.call @hipdnn_ep_op_state_set(%[[STATE]], %[[SLOT]], %[[ST]])
// CHECK: llvm.return
// CHECK: ^[[FAIL]]:
// CHECK: llvm.return

module {
  func.func @one_skip_rms_norm(%ctx: !hip.context) {
    %input = memref.alloc() : memref<128x512xf32, 1>
    %skip = memref.alloc() : memref<128x512xf32, 1>
    %gamma = memref.alloc() : memref<512xf32, 1>
    %output = memref.alloc() : memref<128x512xf32, 1>
    %skip_output = memref.alloc() : memref<128x512xf32, 1>
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma : memref<128x512xf32, 1>, memref<128x512xf32, 1>, memref<512xf32, 1>)
        outs(%output, %skip_output : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
        {epsilon = 9.99999974e-06 : f32}
    return
  }
}
