// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-use-output-allocator --verify-diagnostics %s

module {
  func.func @bad_loop_view(%ctx: !hip.context, %seed: memref<8xf32>)
      -> memref<4xf32, strided<[2]>> {
    %zero = arith.constant 0 : index
    %true = arith.constant true
    %result, %frame = hip.loop(%ctx, %zero, %true)
        iter_args(%seed : memref<8xf32>)
        -> (memref<8xf32>, !hip.loop_frame)
        body @bad_loop_view_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    %view = memref.subview %result[0] [4] [2]
        : memref<8xf32> to memref<4xf32, strided<[2]>>
    // expected-error @below {{'func.return' op loop-backed graph output #0 must be a contiguous identity-compatible ranked memref}}
    return %view : memref<4xf32, strided<[2]>>
  }

  func.func private @bad_loop_view_body(
      %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
      %current: memref<8xf32>, %frame: !hip.loop_frame)
      -> (i32, memref<8xf32>) {
    %status = arith.constant 0 : i32
    return %status, %current : i32, memref<8xf32>
  }
}
