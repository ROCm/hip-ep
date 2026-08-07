// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @constant_negative_axis(%ctx: !hip.context,
                                  %data: tensor<2x3x4xf32>,
                                  %out: tensor<2x3x1xf32>)
    -> tensor<2x3x1xf32> {
  %axes = arith.constant dense<[-1]> : tensor<1xi64>
  %result = hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<2x3x4xf32>, tensor<1xi64>)
    outs(%out : tensor<2x3x1xf32>)
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<2x3x1xf32>
  return %result : tensor<2x3x1xf32>
}

// -----

func.func @constant_drop_axes_rank_zero(%ctx: !hip.context,
                                        %data: tensor<2x3xf32>,
                                        %out: tensor<f32>) -> tensor<f32> {
  %axes = arith.constant dense<[0, 1]> : tensor<2xi64>
  %result = hip.reduce_mean(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<2xi64>)
    outs(%out : tensor<f32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<f32>
  return %result : tensor<f32>
}

// -----

func.func @empty_axes_noop(%ctx: !hip.context,
                           %data: tensor<2x3xf32>,
                           %out: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_max(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<0xi64>)
    outs(%out : tensor<2x3xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 1 : i64}
    : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// -----

func.func @empty_axes_reduce_all(%ctx: !hip.context,
                                 %data: tensor<2x3xf32>,
                                 %out: tensor<1x1xf32>)
    -> tensor<1x1xf32> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_min(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<0xi64>)
    outs(%out : tensor<1x1xf32>)
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// -----

func.func @runtime_axes_outs_fallback(%ctx: !hip.context,
                                      %data: tensor<?x?x?xf32>,
                                      %axes: tensor<?xi64>,
                                      %out: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %result = hip.reduce_prod(%ctx)
    ins(%data, %axes : tensor<?x?x?xf32>, tensor<?xi64>)
    outs(%out : tensor<?x?xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

func.func @rank_zero_data(%ctx: !hip.context,
                          %data: tensor<f32>,
                          %out: tensor<f32>) -> tensor<f32> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_l2(%ctx)
    ins(%data, %axes : tensor<f32>, tensor<0xi64>)
    outs(%out : tensor<f32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<f32>
  return %result : tensor<f32>
}

// -----

func.func @invalid_axis(%ctx: !hip.context,
                        %data: tensor<2x3xf32>,
                        %out: tensor<2xf32>) -> tensor<2xf32> {
  %axes = arith.constant dense<[2]> : tensor<1xi64>
  // expected-error @+1 {{constant axes must be unique and in the range [-rank, rank)}}
  %result = hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<1xi64>)
    outs(%out : tensor<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

func.func @duplicate_axes(%ctx: !hip.context,
                          %data: tensor<2x3xf32>,
                          %out: tensor<2xf32>) -> tensor<2xf32> {
  %axes = arith.constant dense<[1, -1]> : tensor<2xi64>
  // expected-error @+1 {{constant axes must be unique and in the range [-rank, rank)}}
  %result = hip.reduce_mean(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<2xi64>)
    outs(%out : tensor<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

func.func @wrong_constant_output(%ctx: !hip.context,
                                 %data: tensor<2x3x4xf32>,
                                 %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %axes = arith.constant dense<[2]> : tensor<1xi64>
  // expected-error @+1 {{dim 1 of result mismatch: expected 3}}
  %result = hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<2x3x4xf32>, tensor<1xi64>)
    outs(%out : tensor<2x4xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
    : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// -----

func.func @mixed_tensor_memref_mode(%ctx: !hip.context,
                                    %data: tensor<2x3xf32>,
                                    %axes: tensor<1xi64>,
                                    %out: memref<2xf32, 1>) {
  // expected-error @+1 {{all data operands must be the same kind (all tensor or all memref)}}
  hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<1xi64>)
    outs(%out : memref<2xf32, 1>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
  return
}
