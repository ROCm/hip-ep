// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --hip-promote-strided-operands %s

func.func @missing_body(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  // expected-error @below {{body_func 'missing_body_func' does not reference a func.func}}
  %result, %frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @missing_body_func
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}

// -----

func.func private @bad_count_body(
    %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
    %current: memref<4x4xf32>, %frame: !hip.loop_frame)
    -> (i32, memref<4x4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4x4xf32>
}

func.func @bad_count(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  // expected-error @below {{body_func argument count mismatch: expected 6 (context, iter, cond, carriers, captures, frame), got 5}}
  %result, %frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @bad_count_body
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}

// -----

func.func private @bad_rank_body(
    %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
    %current: memref<4x4xf32>, %capture: memref<16xf32>,
    %frame: !hip.loop_frame) -> (i32, memref<4x4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4x4xf32>
}

func.func @bad_rank(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  // expected-error @below {{body_func capture #0 type mismatch}}
  %result, %frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @bad_rank_body
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}

// -----

func.func private @bad_element_body(
    %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
    %current: memref<4x4xf32>, %capture: memref<4x4xf16>,
    %frame: !hip.loop_frame) -> (i32, memref<4x4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4x4xf32>
}

func.func @bad_element(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  // expected-error @below {{body_func capture #0 type mismatch}}
  %result, %frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @bad_element_body
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}

// -----

func.func private @bad_layout_body(
    %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
    %current: memref<4x4xf32>,
    %capture: memref<4x4xf32, strided<[4, 1], offset: 1>>,
    %frame: !hip.loop_frame) -> (i32, memref<4x4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4x4xf32>
}

func.func @bad_layout(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  // expected-error @below {{body_func capture #0 type mismatch}}
  %result, %frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @bad_layout_body
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  return
}
