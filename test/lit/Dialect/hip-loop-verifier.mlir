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
                                 %iter: tensor<i64>,
                                 %cond_in: tensor<i1>,
                                 %v_in: tensor<?x?x?xf16>,
                                 %frame: !hip.loop_frame) -> (i32, tensor<?x?x?xf16>) {
  %status = arith.constant 0 : i32
  return %status, %v_in : i32, tensor<?x?x?xf16>
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
                                      %iter: tensor<i64>,
                                      %cond_in: tensor<i1>,
                                      %v_in: tensor<f16>,
                                      %frame: !hip.loop_frame) -> (i32, tensor<f16>) {
  %status = arith.constant 0 : i32
  return %status, %v_in : i32, tensor<f16>
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
                                   %iter: tensor<i64>,
                                   %cond_in: tensor<i1>,
                                   %a_in: tensor<16xf32>,
                                   %b_in: tensor<?xf32>,
                                   %frame: !hip.loop_frame) -> (i32, tensor<16xf32>, tensor<?xf32>) {
  %status = arith.constant 0 : i32
  return %status, %a_in, %b_in : i32, tensor<16xf32>, tensor<?xf32>
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

// -----

func.func private @loop_body_static_mismatch(
    %ctx: !hip.context, %iter: tensor<i64>, %cond_in: tensor<i1>,
    %v_in: tensor<4x8xf32>, %frame: !hip.loop_frame)
    -> (i32, tensor<4x16xf32>) {
  %status = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<4x16xf32>
  return %status, %empty : i32, tensor<4x16xf32>
}
func.func @loop_static_extent_mismatch(
    %ctx: !hip.context, %M: index, %cond: i1,
    %v: tensor<4x8xf32>) -> tensor<4x16xf32> {
  // expected-error @below {{result type #0 ('tensor<4x16xf32>') must match v_init type #0 ('tensor<4x8xf32>')}}
  %r = hip.loop(%ctx, %M, %cond)
      iter_args(%v : tensor<4x8xf32>)
      -> (tensor<4x16xf32>)
      body @loop_body_static_mismatch
      {num_loop_carried = 1 : i32, cond_is_passthrough}
  return %r : tensor<4x16xf32>
}

// -----

func.func private @loop_body_missing_frame(
    %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
    %current: tensor<?xf32>) -> (i32, tensor<?xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, tensor<?xf32>
}
func.func @loop_missing_frame(%ctx: !hip.context, %M: index, %cond: i1,
                              %seed: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{body_func argument count mismatch: expected 5 (context, iter, cond, carriers, captures, frame), got 4}}
  %result = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : tensor<?xf32>) -> (tensor<?xf32>)
      body @loop_body_missing_frame
      {cond_is_passthrough, num_loop_carried = 1 : i32}
  return %result : tensor<?xf32>
}

// -----

func.func private @loop_body_bad_status(
    %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
    %current: tensor<?xf32>, %frame: !hip.loop_frame)
    -> (i64, tensor<?xf32>) {
  %status = arith.constant 0 : i64
  return %status, %current : i64, tensor<?xf32>
}
func.func @loop_bad_status(%ctx: !hip.context, %M: index, %cond: i1,
                           %seed: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{body_func result #0 must be i32 status}}
  %result = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : tensor<?xf32>) -> (tensor<?xf32>)
      body @loop_body_bad_status
      {cond_is_passthrough, num_loop_carried = 1 : i32}
  return %result : tensor<?xf32>
}

// -----

func.func private @loop_body_bad_capture(
    %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
    %current: tensor<?xf32>, %scalar_capture: i32,
    %frame: !hip.loop_frame) -> (i32, tensor<?xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, tensor<?xf32>
}
func.func @loop_bad_capture(%ctx: !hip.context, %M: index, %cond: i1,
                            %seed: tensor<?xf32>, %capture: i32)
    -> tensor<?xf32> {
  // expected-error @below {{capture #0 must be a lowering-supported ranked tensor; context is threaded separately}}
  %result = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : tensor<?xf32>)
      captures(%capture : i32)
      -> (tensor<?xf32>)
      body @loop_body_bad_capture
      {cond_is_passthrough, num_loop_carried = 1 : i32}
  return %result : tensor<?xf32>
}

// -----

func.func private @loop_body_tensor_mode_memref_iter(
    %ctx: !hip.context, %iter: memref<i64>, %cond: tensor<i1>,
    %current: tensor<?xf32>, %frame: !hip.loop_frame)
    -> (i32, tensor<?xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, tensor<?xf32>
}
func.func @loop_tensor_mode_memref_iter(
    %ctx: !hip.context, %M: index, %cond: i1, %seed: tensor<?xf32>)
    -> tensor<?xf32> {
  // expected-error @below {{body_func argument #1 must be rank-zero i64 ranked tensor iter}}
  %result = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : tensor<?xf32>) -> (tensor<?xf32>)
      body @loop_body_tensor_mode_memref_iter
      {cond_is_passthrough, num_loop_carried = 1 : i32}
  return %result : tensor<?xf32>
}

// -----

func.func private @loop_body_descriptor_tensor_cond(
    %ctx: !hip.context, %iter: memref<i64>, %cond: tensor<i1>,
    %current: memref<4xf32>, %frame: !hip.loop_frame)
    -> (i32, memref<4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4xf32>
}
func.func @loop_descriptor_tensor_cond(
    %ctx: !hip.context, %M: index, %cond: i1, %seed: memref<4xf32>) {
  // expected-error @below {{body_func argument #2 must be rank-zero i1/i8 memref condition}}
  %result, %frame = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : memref<4xf32>)
      -> (memref<4xf32>, !hip.loop_frame)
      body @loop_body_descriptor_tensor_cond
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}

// -----

func.func private @loop_body_tensor_mode_vector_cond(
    %ctx: !hip.context, %iter: tensor<i64>, %cond: vector<i1>,
    %current: tensor<?xf32>, %frame: !hip.loop_frame)
    -> (i32, tensor<?xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, tensor<?xf32>
}
func.func @loop_tensor_mode_vector_cond(
    %ctx: !hip.context, %M: index, %cond: i1, %seed: tensor<?xf32>)
    -> tensor<?xf32> {
  // expected-error @below {{body_func argument #2 must be rank-zero i1/i8 ranked tensor condition}}
  %result = hip.loop(%ctx, %M, %cond)
      iter_args(%seed : tensor<?xf32>) -> (tensor<?xf32>)
      body @loop_body_tensor_mode_vector_cond
      {cond_is_passthrough, num_loop_carried = 1 : i32}
  return %result : tensor<?xf32>
}
