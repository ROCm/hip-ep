// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --assign-op-state-slots gives EACH instance of an op implementing
// OpStateOpInterface its own dense slot (per-instance, not per-class) and
// records the total count as the module attribute hipdnn.num_op_state_slots.
// See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots | FileCheck %s

// Two matmul instances -> two independent slots (0, 1) and a count of 2.
// CHECK: module attributes {{.*}}hipdnn.num_op_state_slots = 2 : i32
module {
  func.func @two_matmuls(
      %ctx: !hip.context,
      %A: memref<1x128x4096xf16, 1>,
      %B: memref<4096x1024xf16, 1>,
      %output: memref<1x128x1024xf16, 1>) {
    // CHECK: hip.matmul
    // CHECK-SAME: hip.op_state_slot = 0 : i32
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x128x4096xf16, 1>, memref<4096x1024xf16, 1>)
        outs(%output : memref<1x128x1024xf16, 1>)

    // CHECK: hip.matmul
    // CHECK-SAME: hip.op_state_slot = 1 : i32
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x128x4096xf16, 1>, memref<4096x1024xf16, 1>)
        outs(%output : memref<1x128x1024xf16, 1>)
    return
  }
}
