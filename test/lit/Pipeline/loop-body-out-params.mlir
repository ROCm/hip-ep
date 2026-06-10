// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-loop-body-to-out-params %s | FileCheck %s --check-prefix=OUTPARAMS
// RUN: hip-mlir-opt --hip-loop-body-to-out-params --convert-hip-to-llvm %s 2>&1 | FileCheck %s --check-prefix=LLVM

// What this file tests
// --------------------
// The `--hip-loop-body-to-out-params` pass (lib/Dialect/Transforms/
// LoopBodyToOutParams.cpp), which promotes private `*_loop_body_*`
// functions from a return-based memref ABI to the out-param ABI that
// `convert-hip-to-llvm` LoopLowering implements:
//
//   - loop-carried result becomes an extra func argument tagged
//     `{bufferize.result}` (`out_param_abi`),
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
    hip.loop(%ctx, %c4, %true) iter_args(%v_carry : memref<16xf32>)
                               captures(%B : memref<16xf32>)
                               body @main_graph_loop_body_n0
                               {cond_is_passthrough, num_loop_carried = 1 : i32}
    return
  }

  // Return-based ABI as left by one-shot-bufferize (one loop-carried memref).
  func.func private @main_graph_loop_body_n0(%ctx: !hip.context,
                                             %iter: memref<i64>,
                                             %cond_in: memref<i1>,
                                             %v_in: memref<16xf32>,
                                             %cap: memref<16xf32>) -> memref<16xf32> {
    %out = memref.alloc() : memref<16xf32>
    hip.add(%ctx) ins(%v_in, %cap : memref<16xf32>, memref<16xf32>)
                  outs(%out : memref<16xf32>)
    return %out : memref<16xf32>
  }
}

// OUTPARAMS-LABEL: func.func private @main_graph_loop_body_n0
// OUTPARAMS-SAME: %{{.*}}: memref<16xf32> {bufferize.result})
// OUTPARAMS-NOT: -> memref
// OUTPARAMS: return

// LLVM-LABEL: llvm.func internal @main_graph_loop_body_n0_trampoline
// LLVM: llvm.call @main_graph_loop_body_n0(
