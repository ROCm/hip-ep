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
                        %v_carry: memref<16xf32>,
                        %scan_input: memref<2x3xi64>,
                        %scan_state: memref<i64>,
                        %scan_output: memref<2x4xi64>) {
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    memref.copy %A, %v_carry : memref<16xf32> to memref<16xf32>
    hip.loop(%ctx, %c4, %true) iter_args(%v_carry : memref<16xf32>)
                               captures(%B : memref<16xf32>)
                               body @main_graph_loop_body_n0
                               {cond_is_passthrough, num_loop_carried = 1 : i32}
    hip.scan(%ctx) scan_inputs(%scan_input : memref<2x3xi64>)
                    outs(%scan_state, %scan_output
                         : memref<i64>, memref<2x4xi64>)
                    body @main_graph_scan_body_n0
                    {num_scan_inputs = 1 : i32,
                     num_scan_outputs = 1 : i32,
                     num_state_variables = 1 : i32}
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

  func.func private @main_graph_scan_body_n0(
      %ctx: !hip.context, %state_in: memref<i64>,
      %slice: memref<3xi64>) -> (memref<i64>, memref<4xi64>)
      attributes {hip.scan_num_state_variables = 1 : i32} {
    %next_state = memref.alloc() : memref<i64>
    %scan_value = memref.alloc() : memref<4xi64>
    return %next_state, %scan_value : memref<i64>, memref<4xi64>
  }
}

// OUTPARAMS-LABEL: func.func @main_graph
// OUTPARAMS: %[[CALLER_SCRATCH:.*]] = memref.alloc() : memref<i64>
// OUTPARAMS: hip.scan
// OUTPARAMS-SAME: state_scratch(%[[CALLER_SCRATCH]] : memref<i64>)

// OUTPARAMS-LABEL: func.func private @main_graph_loop_body_n0
// OUTPARAMS-SAME: %{{.*}}: memref<16xf32> {bufferize.result})
// OUTPARAMS-NOT: -> memref
// OUTPARAMS: return

// Scan state computation uses caller-owned scratch and is copied to the state
// out-param at the terminator. The ordinary scan-output allocation is hoisted
// directly into its out-param.
// OUTPARAMS-LABEL: func.func private @main_graph_scan_body_n0
// OUTPARAMS-SAME: %[[STATE_SCRATCH:.*]]: memref<i64>
// OUTPARAMS-SAME: %[[STATE_OUT:.*]]: memref<i64> {bufferize.result}
// OUTPARAMS-SAME: %[[SCAN_OUT:.*]]: memref<4xi64> {bufferize.result})
// OUTPARAMS-NOT: memref.alloc() : memref<i64>
// OUTPARAMS-NOT: memref.alloc() : memref<4xi64>
// OUTPARAMS: memref.copy %[[STATE_SCRATCH]], %[[STATE_OUT]]
// OUTPARAMS-NOT: memref.copy {{.*}}, %[[SCAN_OUT]]
// OUTPARAMS: return

// LLVM-LABEL: llvm.func internal @main_graph_loop_body_n0_trampoline
// LLVM: llvm.call @main_graph_loop_body_n0(
