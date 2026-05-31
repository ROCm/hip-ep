// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Pin the `hip.loop` verifier contract that result_type[i] must equal
// v_init[i].getType() (HipDialect.cpp::LoopOp::verify, HipOps.td:126-127).
// This is also the contract that `LoopOp::inferReturnTypes`
// (InferTypeOpInterface) is engineered to preserve when a future caller
// constructs a `hip.loop` via the InferType-aware builder without
// supplying explicit result types -- the InferType impl simply returns
// v_init operand types per result, which by definition agrees with this
// verifier check.
//
// Coverage:
//   1. Positive: refined dynamic v_init (rank-3 with `?` dims) +
//      matching result type verifies cleanly.
//   2. Negative: under-refined result type (rank-0) against refined
//      rank-3 dynamic v_init triggers the canonical
//      "result type #N must match v_init type #N" error -- this is the
//      signature seen on dynamic-shape vision-encoder ONNX exports
//      where the importer leaves loop result types under-refined
//      relative to the v_init operands. The fix lives in PR #265
//      (LoopOutline.cpp body refinement + InferType builder migration);
//      this LIT pins the verifier so the fix can't silently regress.
//   3. Positive: multi-v_init (mixed static + dynamic) verifies cleanly.
// ============================================================================

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @loop_dynamic_v_init_ok
// CHECK:         hip.loop
func.func private @loop_body_dyn(%ctx: !hip.context,
                                 %iter: index,
                                 %cond_in: i1,
                                 %v_in: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  return %v_in : tensor<?x?x?xf16>
}
func.func @loop_dynamic_v_init_ok(%ctx: !hip.context,
                                  %M: index,
                                  %cond: i1,
                                  %v: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%v : tensor<?x?x?xf16>)
                 -> (tensor<?x?x?xf16>)
                 body @loop_body_dyn
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<?x?x?xf16>
}

// -----

// Negative case -- canonical Qwen3.5-9B-vision-style mismatch: result
// type is rank-0 (under-refined), v_init is rank-3 dynamic (refined).
// `LoopOp::inferReturnTypes` would have produced rank-3 here; manually
// supplying rank-0 is what the historical bug looked like before the
// fix.
func.func private @loop_body_mismatch(%ctx: !hip.context,
                                      %iter: index,
                                      %cond_in: i1,
                                      %v_in: tensor<f16>) -> tensor<f16> {
  return %v_in : tensor<f16>
}
func.func @loop_v_init_result_mismatch(%ctx: !hip.context,
                                       %M: index,
                                       %cond: i1,
                                       %v: tensor<?x?x?xf16>) -> tensor<f16> {
  // expected-error @below {{result type #0 ('tensor<f16>') must match v_init type #0 ('tensor<?x?x?xf16>')}}
  %r = hip.loop(%ctx, %M, %cond)
                 iter_args(%v : tensor<?x?x?xf16>)
                 -> (tensor<f16>)
                 body @loop_body_mismatch
                 {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<f16>
}

// -----

// CHECK-LABEL: func.func @loop_multi_v_init_ok
// CHECK:         hip.loop
func.func private @loop_body_multi(%ctx: !hip.context,
                                   %iter: index,
                                   %cond_in: i1,
                                   %a_in: tensor<16xf32>,
                                   %b_in: tensor<?xf32>) -> (tensor<16xf32>, tensor<?xf32>) {
  return %a_in, %b_in : tensor<16xf32>, tensor<?xf32>
}
func.func @loop_multi_v_init_ok(%ctx: !hip.context,
                                %M: index,
                                %cond: i1,
                                %a: tensor<16xf32>,
                                %b: tensor<?xf32>) -> (tensor<16xf32>, tensor<?xf32>) {
  %r:2 = hip.loop(%ctx, %M, %cond)
                   iter_args(%a, %b : tensor<16xf32>, tensor<?xf32>)
                   -> (tensor<16xf32>, tensor<?xf32>)
                   body @loop_body_multi
                   {num_loop_carried = 2 : i32, cond_is_passthrough}
  return %r#0, %r#1 : tensor<16xf32>, tensor<?xf32>
}
