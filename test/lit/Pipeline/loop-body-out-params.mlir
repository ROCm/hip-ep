// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-loop-body-to-out-params %s | FileCheck %s --check-prefix=OUTPARAMS
// RUN: hip-mlir-opt --hip-loop-body-to-out-params --convert-hip-to-llvm %s 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: hip-mlir-opt --hip-loop-body-to-out-params --hip-pool-allocs %s | FileCheck %s --check-prefix=POOL
// RUN: hip-mlir-opt --hip-loop-body-to-out-params --hip-pool-allocs --hip-prepare-loop-body-failures %s | FileCheck %s --check-prefix=FAILFAST

// What this file tests
// --------------------
// The `--hip-loop-body-to-out-params` pass (lib/Dialect/Transforms/
// LoopBodyToOutParams.cpp), which preserves private `*_loop_body_*`
// status + descriptor results and redirects carrier storage to the
// per-invocation frame ABI:
//
//   - loop-carried results stay descriptor returns and their allocations become
//     frame-owned `hip.loop_alloc` operations,
//   - the pass composes with `--convert-hip-to-llvm` without ABI mismatch
//     (`llvm_lowering`).

module {
  func.func @main_graph(%ctx: !hip.context,
                        %A: memref<16xf32>,
                        %B: memref<16xf32>,
                        %v_carry: memref<16xf32>) {
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    memref.copy %A, %v_carry : memref<16xf32> to memref<16xf32>
    %result, %loop_frame = hip.loop(%ctx, %c4, %true) iter_args(%v_carry : memref<16xf32>)
                               captures(%B : memref<16xf32>)
                               -> (memref<16xf32>, !hip.loop_frame)
                               body @main_graph_loop_body_n0
                               {cond_is_passthrough, descriptor_return, num_loop_carried = 1 : i32}
    return
  }

  // Return-based ABI as left by one-shot-bufferize (one loop-carried memref).
  func.func private @main_graph_loop_body_n0(%ctx: !hip.context,
                                             %iter: memref<i64>,
                                             %cond_in: memref<i1>,
                                             %v_in: memref<16xf32>,
                                             %cap: memref<16xf32>,
                                             %frame: !hip.loop_frame) -> (i32, memref<16xf32>) {
    %status = arith.constant 0 : i32
    %tmp = memref.alloc() : memref<16xf32>
    memref.copy %cap, %tmp : memref<16xf32> to memref<16xf32>
    %out = memref.alloc() : memref<16xf32>
    hip.add(%ctx) ins(%v_in, %tmp : memref<16xf32>, memref<16xf32>)
                  outs(%out : memref<16xf32>)
    return %status, %out : i32, memref<16xf32>
  }
}

// OUTPARAMS-LABEL: func.func private @main_graph_loop_body_n0
// OUTPARAMS-SAME: %[[FRAME:[^ ,]+]]: !hip.loop_frame) -> (i32, memref<16xf32>)
// OUTPARAMS: %[[OUT:.*]] = hip.loop_alloc(%[[FRAME]]) {carrier_index = 0 : i32} : memref<16xf32>
// OUTPARAMS: return %{{.*}}, %[[OUT]] : i32, memref<16xf32>

// POOL-LABEL: func.func private @main_graph_loop_body_n0
// POOL: hip.get_pool
// POOL: hip.loop_alloc

// FAILFAST-LABEL: func.func private @main_graph_loop_body_n0
// FAILFAST: hip.loop_alloc
// FAILFAST-NEXT: hip.loop_frame_status
// FAILFAST: cf.cond_br
// FAILFAST: ^bb{{.*}}:
// FAILFAST: hip.add
// FAILFAST: ^bb{{.*}}(%[[STATUS:.*]]: i32):
// FAILFAST: return %[[STATUS]], %{{.*}} : i32, memref<16xf32>

// LLVM-LABEL: llvm.func internal @main_graph_loop_body_n0_trampoline
// LLVM: llvm.call @main_graph_loop_body_n0(
