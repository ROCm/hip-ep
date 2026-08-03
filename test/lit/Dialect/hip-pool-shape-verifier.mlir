// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// Signed ceil division matters here: ceil((1 - 2) / 2) + 1 = 1.
func.func @valid_ceil_mode(
    %ctx: !hip.context,
    %input: memref<1x3x1xf32, 1>,
    %output: memref<1x3x1xf32, 1>) {
  hip.pool(%ctx) ins(%input : memref<1x3x1xf32, 1>)
    outs(%output : memref<1x3x1xf32, 1>)
    {pool_mode = 0, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

func.func @wrong_values_shape(
    %ctx: !hip.context,
    %input: memref<1x3x8x8xf32, 1>,
    %output: memref<1x3x5x4xf32, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 4}}
  hip.pool(%ctx) ins(%input : memref<1x3x8x8xf32, 1>)
    outs(%output : memref<1x3x5x4xf32, 1>)
    {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
     pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0,
     storage_order = 0}
  return
}

// -----

func.func @wrong_indices_shape(
    %ctx: !hip.context,
    %input: memref<1x3x8x8xf32, 1>,
    %values: memref<1x3x4x4xf32, 1>,
    %indices: memref<1x3x4x5xi64, 1>) {
  // expected-error @+1 {{dim 3 of result mismatch: expected 4}}
  hip.pool(%ctx) ins(%input : memref<1x3x8x8xf32, 1>)
    outs(%values, %indices : memref<1x3x4x4xf32, 1>,
                             memref<1x3x4x5xi64, 1>)
    {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
     pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0,
     storage_order = 0}
  return
}

// -----

func.func @indices_on_average(
    %ctx: !hip.context,
    %input: memref<1x3x8xf32, 1>,
    %values: memref<1x3x4xf32, 1>,
    %indices: memref<1x3x4xi64, 1>) {
  // expected-error @+1 {{only max pooling may produce an indices output}}
  hip.pool(%ctx) ins(%input : memref<1x3x8xf32, 1>)
    outs(%values, %indices : memref<1x3x4xf32, 1>,
                             memref<1x3x4xi64, 1>)
    {pool_mode = 0, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @wrong_indices_type(
    %ctx: !hip.context,
    %input: memref<1x3x8xf32, 1>,
    %values: memref<1x3x4xf32, 1>,
    %indices: memref<1x3x4xi32, 1>) {
  // expected-error @+1 {{indices output must have i64 element type}}
  hip.pool(%ctx) ins(%input : memref<1x3x8xf32, 1>)
    outs(%values, %indices : memref<1x3x4xf32, 1>,
                             memref<1x3x4xi32, 1>)
    {pool_mode = 1, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @invalid_pool_mode(
    %ctx: !hip.context,
    %input: memref<1x3x8xf32, 1>,
    %output: memref<1x3x4xf32, 1>) {
  // expected-error @+1 {{pool_mode must be 0}}
  hip.pool(%ctx) ins(%input : memref<1x3x8xf32, 1>)
    outs(%output : memref<1x3x4xf32, 1>)
    {pool_mode = 3, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @wrong_values_type(
    %ctx: !hip.context,
    %input: memref<1x3x8xf32, 1>,
    %output: memref<1x3x4xf16, 1>) {
  // expected-error @+1 {{input and values output must have the same floating-point type}}
  hip.pool(%ctx) ins(%input : memref<1x3x8xf32, 1>)
    outs(%output : memref<1x3x4xf16, 1>)
    {pool_mode = 1, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @window_product_overflow(
    %ctx: !hip.context,
    %input: memref<1x1x?xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  // expected-error @+1 {{pool window parameters overflow index arithmetic at spatial axis 0}}
  hip.pool(%ctx) ins(%input : memref<1x1x?xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {pool_mode = 0, kernel_shape = [9223372036854775807], strides = [1],
     pads = [0, 0], dilations = [9223372036854775807], ceil_mode = 0,
     storage_order = 0}
  return
}

// -----

func.func @window_offset_i64_boundary(
    %ctx: !hip.context,
    %input: memref<1x1x?xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  hip.pool(%ctx) ins(%input : memref<1x1x?xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {pool_mode = 0, kernel_shape = [9223372036854775807], strides = [1],
     pads = [9223372036854775807, 9223372036854775807],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}
