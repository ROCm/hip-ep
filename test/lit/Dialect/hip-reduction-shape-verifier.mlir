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
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 2>}
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
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0, 1>}
    : tensor<f32>
  return %result : tensor<f32>
}

// -----

func.func @empty_axes_noop(%ctx: !hip.context,
                           %data: tensor<2x3xf16>,
                           %out: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_max(%ctx)
    ins(%data, %axes : tensor<2x3xf16>, tensor<0xi64>)
    outs(%out : tensor<2x3xf16>)
    {keepdims = 0 : i64, noop_with_empty_axes = 1 : i64,
     normalized_axes = array<i64>}
    : tensor<2x3xf16>
  return %result : tensor<2x3xf16>
}

// -----

func.func @empty_axes_reduce_all(%ctx: !hip.context,
                                 %data: tensor<2x3xi32>,
                                 %out: tensor<1x1xi32>)
    -> tensor<1x1xi32> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_min(%ctx)
    ins(%data, %axes : tensor<2x3xi32>, tensor<0xi64>)
    outs(%out : tensor<1x1xi32>)
    {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0, 1>}
    : tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}

// -----

func.func @runtime_axes_dynamic_out_rejected(%ctx: !hip.context,
                                             %data: tensor<?x?x?xi32>,
                                             %axes: tensor<?xi64>,
                                             %out: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
  // expected-error @+1 {{axes must have a structurally-proven compile-time constant source}}
  %result = hip.reduce_prod(%ctx)
    ins(%data, %axes : tensor<?x?x?xi32>, tensor<?xi64>)
    outs(%out : tensor<?x?xi32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0>}
    : tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// -----

func.func @runtime_axes_static_out_rejected(%ctx: !hip.context,
                                            %data: tensor<2x3xi32>,
                                            %axes: tensor<1xi64>,
                                            %out: tensor<2xi32>)
    -> tensor<2xi32> {
  // expected-error @+1 {{axes must have a structurally-proven compile-time constant source}}
  %result = hip.reduce_prod(%ctx)
    ins(%data, %axes : tensor<2x3xi32>, tensor<1xi64>)
    outs(%out : tensor<2xi32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<2xi32>
  return %result : tensor<2xi32>
}

// -----

func.func @noncontiguous_axes_rejected(%ctx: !hip.context,
                                       %data: tensor<2x3x4xi32>,
                                       %out: tensor<3xi32>) -> tensor<3xi32> {
  %axes = arith.constant dense<[0, 2]> : tensor<2xi64>
  // expected-error @+1 {{constant axes must be unique, in range, and form one contiguous span}}
  %result = hip.reduce_max(%ctx)
    ins(%data, %axes : tensor<2x3x4xi32>, tensor<2xi64>)
    outs(%out : tensor<3xi32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0, 2>}
    : tensor<3xi32>
  return %result : tensor<3xi32>
}

// -----

memref.global "private" constant @axes1 : memref<1xi64> = dense<[1]>

func.func @memref_constant_axes(%ctx: !hip.context,
                                %data: memref<2x3xi32>,
                                %out: memref<2xi32>) {
  %axes = memref.get_global @axes1 : memref<1xi64>
  hip.reduce_prod(%ctx)
    ins(%data, %axes : memref<2x3xi32>, memref<1xi64>)
    outs(%out : memref<2xi32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
  return
}

// -----

memref.global "private" constant @axes1 : memref<1xi64> = dense<[1]>

func.func @forged_normalized_axes_rejected(%ctx: !hip.context,
                                           %data: memref<2x3xf32>,
                                           %out: memref<2xf32>) {
  %axes = memref.get_global @axes1 : memref<1xi64>
  // expected-error @+1 {{normalized_axes must exactly match the normalized constant axes source}}
  hip.reduce_sum(%ctx)
    ins(%data, %axes : memref<2x3xf32>, memref<1xi64>)
    outs(%out : memref<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0>}
  return
}

// -----

func.func @memref_runtime_axes_rejected(%ctx: !hip.context,
                                        %data: memref<2x3xi32>,
                                        %axes: memref<1xi64>,
                                        %out: memref<2xi32>) {
  // expected-error @+1 {{axes must have a structurally-proven compile-time constant source}}
  hip.reduce_min(%ctx)
    ins(%data, %axes : memref<2x3xi32>, memref<1xi64>)
    outs(%out : memref<2xi32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
  return
}

// -----

memref.global "private" constant @axes02 : memref<2xi64> = dense<[0, 2]>

func.func @memref_noncontiguous_axes_rejected(
    %ctx: !hip.context, %data: memref<2x3x4xf32>,
    %out: memref<3xf32>) {
  %axes = memref.get_global @axes02 : memref<2xi64>
  // expected-error @+1 {{constant axes must be unique, in range, and form one contiguous span}}
  hip.reduce_l2(%ctx)
    ins(%data, %axes : memref<2x3x4xf32>, memref<2xi64>)
    outs(%out : memref<3xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 0, 2>}
  return
}

// -----

func.func @rank_zero_data(%ctx: !hip.context,
                          %data: tensor<f32>,
                          %out: tensor<f32>) -> tensor<f32> {
  %axes = arith.constant dense<> : tensor<0xi64>
  %result = hip.reduce_l2(%ctx)
    ins(%data, %axes : tensor<f32>, tensor<0xi64>)
    outs(%out : tensor<f32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64>}
    : tensor<f32>
  return %result : tensor<f32>
}

// -----

func.func @invalid_axis(%ctx: !hip.context,
                        %data: tensor<2x3xf32>,
                        %out: tensor<2xf32>) -> tensor<2xf32> {
  %axes = arith.constant dense<[2]> : tensor<1xi64>
  // expected-error @+1 {{constant axes must be unique, in range, and form one contiguous span}}
  %result = hip.reduce_sum(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<1xi64>)
    outs(%out : tensor<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

func.func @duplicate_axes(%ctx: !hip.context,
                          %data: tensor<2x3xf32>,
                          %out: tensor<2xf32>) -> tensor<2xf32> {
  %axes = arith.constant dense<[1, -1]> : tensor<2xi64>
  // expected-error @+1 {{constant axes must be unique, in range, and form one contiguous span}}
  %result = hip.reduce_mean(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<2xi64>)
    outs(%out : tensor<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
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
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 2>}
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
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
  return
}

// -----

func.func @reduce_max_f32_rejected(%ctx: !hip.context,
                                   %data: tensor<2x3xf32>,
                                   %out: tensor<2xf32>) -> tensor<2xf32> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  // expected-error @+1 {{unsupported reduction element type 'f32'; supported types: f16, i32, i64}}
  %result = hip.reduce_max(%ctx)
    ins(%data, %axes : tensor<2x3xf32>, tensor<1xi64>)
    outs(%out : tensor<2xf32>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<2xf32>
  return %result : tensor<2xf32>
}

// -----

func.func @reduce_min_bf16_rejected(%ctx: !hip.context,
                                    %data: tensor<2x3xbf16>,
                                    %out: tensor<2xbf16>) -> tensor<2xbf16> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  // expected-error @+1 {{unsupported reduction element type 'bf16'; supported types: f16, i32, i64}}
  %result = hip.reduce_min(%ctx)
    ins(%data, %axes : tensor<2x3xbf16>, tensor<1xi64>)
    outs(%out : tensor<2xbf16>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// -----

func.func @reduce_prod_f64_rejected(%ctx: !hip.context,
                                    %data: tensor<2x3xf64>,
                                    %out: tensor<2xf64>) -> tensor<2xf64> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  // expected-error @+1 {{unsupported reduction element type 'f64'; supported types: f16, i32, i64}}
  %result = hip.reduce_prod(%ctx)
    ins(%data, %axes : tensor<2x3xf64>, tensor<1xi64>)
    outs(%out : tensor<2xf64>)
    {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
     normalized_axes = array<i64: 1>}
    : tensor<2xf64>
  return %result : tensor<2xf64>
}
