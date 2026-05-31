// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pin the `hip.loop` verifier contract that `result_type[i]` must equal
// `v_init[i].getType()`. This is also the contract that
// `LoopOp::inferReturnTypes` is engineered to preserve — the InferType
// impl returns v_init operand types per result, which by definition
// agrees with the verifier.

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

// Negative case: result type is rank-0 (under-refined), v_init is rank-3
// dynamic (refined). `LoopOp::inferReturnTypes` would have produced
// rank-3 here; manually supplying rank-0 is the historical bug pattern
// (the importer leaves loop result types as rank-0 placeholders while
// upstream shape inference has already refined v_init).
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
