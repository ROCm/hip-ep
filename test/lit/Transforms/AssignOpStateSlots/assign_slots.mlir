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

// Two conv instances -> two independent slots (0, 1) and a count of 2.
// CHECK: module attributes {{.*}}hipdnn.num_op_state_slots = 2 : i32
module {
  func.func @two_convs(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // CHECK: hip.conv
    // CHECK-SAME: hip.op_state_slot = 0 : i32
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}

    // CHECK: hip.conv
    // CHECK-SAME: hip.op_state_slot = 1 : i32
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}
    return
  }
}
