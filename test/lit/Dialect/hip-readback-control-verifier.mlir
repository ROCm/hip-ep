// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @dynamic_length(%ctx: !hip.context, %source: tensor<?xi32>) {
  // expected-error @+1 {{requires one or more statically-sized rank-0/rank-1 i32/i64 sources}}
  %r:2 = hip.readback_control(%ctx, %source : tensor<?xi32>) -> (i1, i64)
  return
}

// -----

func.func @rank_two(%ctx: !hip.context, %source: tensor<1x1xi64>) {
  // expected-error @+1 {{requires one or more statically-sized rank-0/rank-1 i32/i64 sources}}
  %r:2 = hip.readback_control(%ctx, %source : tensor<1x1xi64>) -> (i1, i64)
  return
}

// -----

func.func @wrong_element_type(%ctx: !hip.context, %source: tensor<1xf32>) {
  // expected-error @+1 {{requires one or more statically-sized rank-0/rank-1 i32/i64 sources}}
  %r:2 = hip.readback_control(%ctx, %source : tensor<1xf32>) -> (i1, i64)
  return
}

// -----

func.func @wrong_result_count(%ctx: !hip.context, %source: tensor<2xi64>) {
  // expected-error @+1 {{expected 2 i64 value results, got 1}}
  %r:2 = hip.readback_control(%ctx, %source : tensor<2xi64>) -> (i1, i64)
  return
}

// -----

func.func @noncontiguous(%ctx: !hip.context,
                         %source: memref<2xi64, strided<[2]>>) {
  // expected-error @+1 {{memref source #0 must be contiguous}}
  %r:3 = hip.readback_control(
      %ctx, %source : memref<2xi64, strided<[2]>>) -> (i1, i64, i64)
  return
}
