// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

module {
  func.func @bad_alloc_index(%ctx: !hip.context, %seed: memref<4xf32>) {
    %zero = arith.constant 0 : index
    %true = arith.constant true
    %result, %frame = hip.loop(%ctx, %zero, %true)
        iter_args(%seed : memref<4xf32>)
        -> (memref<4xf32>, !hip.loop_frame)
        body @bad_alloc_index_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    return
  }

  func.func private @bad_alloc_index_body(
      %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
      %current: memref<4xf32>, %frame: !hip.loop_frame)
      -> (i32, memref<4xf32>) {
    %status = arith.constant 0 : i32
    // expected-error @below {{'hip.loop_alloc' op carrier_index 2 is outside [0, 1)}}
    %next = hip.loop_alloc(%frame) {carrier_index = 2 : i32} : memref<4xf32>
    return %status, %next : i32, memref<4xf32>
  }
}
