// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the conv lowering threads its compiler-assigned op-state slot
// (hip.op_state_slot, set by --assign-op-state-slots) into the
// wrap_miopenConvolutionForward call as the trailing i32 argument. At runtime
// that slot is how the conv reaches its per-instance ConvState (workspace),
// replacing the old shared RuntimeState::conv_scratch.
// See docs/design/op-state-slots-design.md.
// ============================================================================

// RUN: hip-mlir-opt %s --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s

// One conv -> one slot.
// CHECK: hipdnn.num_op_state_slots = 1 : i32

// Runtime signature gains a trailing i32 (op_state_slot).
// CHECK: llvm.func @wrap_miopenConvolutionForward({{.*}}, i32) -> i32

// The conv call passes slot 0 as its trailing argument.
// CHECK: %[[SLOT:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wrap_miopenConvolutionForward({{.*}}, %[[SLOT]]) :

module {
  func.func @conv_slot(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}
    return
  }
}
