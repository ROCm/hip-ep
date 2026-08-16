// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s

// The zero-trip MAYBE alias from loop result to v_init extends the seed
// allocation lifetime through result consumers. The equally sized post-loop
// allocation therefore cannot reuse offset zero.
// CHECK-LABEL: func.func @seed_liveness
// CHECK: hip.get_pool
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[OFFSET:.*]] = arith.constant 256 : index
// CHECK-NEXT: %[[SEED:.*]] = memref.view %{{.*}}[%[[ZERO]]]
// CHECK: %[[LOOP:[^: ]+]]:2 = hip.loop
// CHECK: %[[AFTER:.*]] = memref.view %{{.*}}[%[[OFFSET]]]
// CHECK: memref.copy %[[LOOP]]#0, %[[AFTER]]
module {
  func.func @seed_liveness(%ctx: !hip.context) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0 : index
    %true = arith.constant true
    %seed = memref.alloc() : memref<16xf32>
    %result, %frame = hip.loop(%ctx, %zero, %true)
        iter_args(%seed : memref<16xf32>)
        -> (memref<16xf32>, !hip.loop_frame)
        body @seed_liveness_body
        {cond_is_passthrough, descriptor_return,
         num_loop_carried = 1 : i32}
    %after = memref.alloc() : memref<16xf32>
    memref.copy %result, %after : memref<16xf32> to memref<16xf32>
    hip.loop_frame_destroy(%ctx, %frame)
    return
  }

  func.func private @seed_liveness_body(
      %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
      %current: memref<16xf32>, %frame: !hip.loop_frame)
      -> (i32, memref<16xf32>) {
    %status = arith.constant 0 : i32
    return %status, %current : i32, memref<16xf32>
  }
}
