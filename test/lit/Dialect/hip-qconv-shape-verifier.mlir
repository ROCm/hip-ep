// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @mixed_bias(%ctx: !hip.context,
    %input: tensor<1x8x1x2xui16>, %weights: tensor<4x8x1x1xi8>,
    %scales: tensor<4xf32>, %zp: tensor<4xi8>,
    %bias: memref<4xf32, 1>, %out: tensor<1x4x1x2xui16>) {
  // expected-error @+1 {{all data operands must be the same kind}}
  %r = hip.qconv(%ctx)
    ins(%input, %weights, %scales, %zp, %bias :
        tensor<1x8x1x2xui16>, tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>, memref<4xf32, 1>)
    outs(%out : tensor<1x4x1x2xui16>)
    {input_scale = 0.1 : f32, input_zp = 3 : i64,
     output_scale = 0.2 : f32, output_zp = 7 : i64,
     weight_axis = 0 : i64, kernel_shape = [1, 1],
     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
     group = 1 : i64, packed_int4} : tensor<1x4x1x2xui16>
  return
}

// -----

func.func @result_init_mismatch(%ctx: !hip.context,
    %input: tensor<1x8x1x2xui16>, %weights: tensor<4x8x1x1xi8>,
    %scales: tensor<4xf32>, %zp: tensor<4xi8>,
    %out: tensor<1x4x1x2xui16>) {
  // expected-error @+1 {{must match DPS init type #0}}
  %r = hip.qconv(%ctx)
    ins(%input, %weights, %scales, %zp :
        tensor<1x8x1x2xui16>, tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>)
    outs(%out : tensor<1x4x1x2xui16>)
    {input_scale = 0.1 : f32, input_zp = 3 : i64,
     output_scale = 0.2 : f32, output_zp = 7 : i64,
     weight_axis = 0 : i64, kernel_shape = [1, 1],
     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
     group = 1 : i64, packed_int4} : tensor<1x4x1x3xui16>
  return
}

// -----

func.func @wrong_channels(%ctx: !hip.context,
    %input: tensor<1x8x1x2xui16>, %weights: tensor<4x8x1x1xi8>,
    %scales: tensor<4xf32>, %zp: tensor<4xi8>,
    %out: tensor<1x8x1x2xui16>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 4}}
  %r = hip.qconv(%ctx)
    ins(%input, %weights, %scales, %zp :
        tensor<1x8x1x2xui16>, tensor<4x8x1x1xi8>, tensor<4xf32>, tensor<4xi8>)
    outs(%out : tensor<1x8x1x2xui16>)
    {input_scale = 0.1 : f32, input_zp = 3 : i64,
     output_scale = 0.2 : f32, output_zp = 7 : i64,
     weight_axis = 0 : i64, kernel_shape = [1, 1],
     strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1],
     group = 1 : i64, packed_int4} : tensor<1x8x1x2xui16>
  return
}

// -----
