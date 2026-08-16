// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize --cse %s | FileCheck %s

module {
  func.func @status_effects(%ctx: !hip.context, %seed: memref<4xf32>) {
    %zero = arith.constant 0 : index
    %true = arith.constant true
    %result, %frame = hip.loop(%ctx, %zero, %true)
        iter_args(%seed : memref<4xf32>)
        -> (memref<4xf32>, !hip.loop_frame)
        body @status_effects_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    hip.loop_frame_destroy(%ctx, %frame)
    return
  }

  // CHECK-LABEL: func.func private @status_effects_body
  // CHECK: hip.loop_frame_status
  // CHECK: hip.loop_alloc
  // CHECK: hip.loop_frame_status
  func.func private @status_effects_body(
      %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
      %current: memref<4xf32>, %frame: !hip.loop_frame)
      -> (i32, memref<4xf32>) {
    %before = hip.loop_frame_status(%frame)
    %next = hip.loop_alloc(%frame) {carrier_index = 0 : i32} : memref<4xf32>
    %after = hip.loop_frame_status(%frame)
    %combined = arith.addi %before, %after : i32
    return %combined, %next : i32, memref<4xf32>
  }
}
