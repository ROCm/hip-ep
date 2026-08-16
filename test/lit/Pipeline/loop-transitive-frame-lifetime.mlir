// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-finalize-loop-frames %s | FileCheck %s

module {
  // A may be the storage returned by B when B takes its zero-trip/pass-through
  // path. A's frame token must therefore survive through B's result consumer.
  // CHECK-LABEL: func.func @sequential_loops
  // CHECK: %[[A:[^: ]+]]:2 = hip.loop
  // CHECK-NOT: hip.loop_frame_destroy
  // CHECK: %[[B:[^: ]+]]:2 = hip.loop
  // CHECK-NOT: hip.loop_frame_destroy
  // CHECK: memref.copy %[[B]]#0
  // CHECK: hip.loop_frame_destroy
  // CHECK: hip.loop_frame_destroy
  func.func @sequential_loops(%ctx: !hip.context, %seed: memref<8xf32>,
                              %out: memref<8xf32>) {
    %zero = arith.constant 0 : index
    %true = arith.constant true
    %a, %fa = hip.loop(%ctx, %zero, %true)
        iter_args(%seed : memref<8xf32>)
        -> (memref<8xf32>, !hip.loop_frame)
        body @pass_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    %b, %fb = hip.loop(%ctx, %zero, %true)
        iter_args(%a : memref<8xf32>)
        -> (memref<8xf32>, !hip.loop_frame)
        body @pass_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    memref.copy %b, %out : memref<8xf32> to memref<8xf32>
    return
  }

  func.func private @pass_body(
      %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
      %current: memref<8xf32>, %frame: !hip.loop_frame)
      -> (i32, memref<8xf32>) {
    %status = arith.constant 0 : i32
    return %status, %current : i32, memref<8xf32>
  }
}
