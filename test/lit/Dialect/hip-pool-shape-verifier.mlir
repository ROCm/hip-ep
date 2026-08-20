// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// Signed ceil division matters here: ceil((1 - 2) / 2) + 1 = 1.
func.func @valid_ceil_mode(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x1xf32, 1>,
    %output: memref<1x3x1xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x1xf32, 1>)
    outs(%output : memref<1x3x1xf32, 1>)
    {pool_mode = 0, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

// The raw ceil formula produces 3, but ONNX drops the last window because its
// start, 4, is at input + pad_begin.
func.func @valid_ceil_trailing_window_rank1_with_indices(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x4xf32, 1>,
    %values: memref<1x3x2xf32, 1>,
    %indices: memref<1x3x2xi64, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x4xf32, 1>)
    outs(%values, %indices : memref<1x3x2xf32, 1>,
                             memref<1x3x2xi64, 1>)
    {pool_mode = 1, kernel_shape = [3], strides = [2], pads = [0, 2],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

func.func @valid_ceil_trailing_window_rank2(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x2x4x4xf32, 1>,
    %output: memref<1x2x2x2xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x2x4x4xf32, 1>)
    outs(%output : memref<1x2x2x2xf32, 1>)
    {pool_mode = 0, kernel_shape = [3, 3], strides = [2, 2],
     pads = [0, 0, 2, 2], dilations = [1, 1], ceil_mode = 1,
     storage_order = 0}
  return
}

// -----

func.func @valid_ceil_trailing_window_rank3(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x2x4x4x4xf32, 1>,
    %output: memref<1x2x2x2x2xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x2x4x4x4xf32, 1>)
    outs(%output : memref<1x2x2x2x2xf32, 1>)
    {pool_mode = 2, kernel_shape = [3, 3, 3], strides = [2, 2, 2],
     pads = [0, 0, 0, 2, 2, 2], dilations = [1, 1, 1], ceil_mode = 1,
     storage_order = 0}
  return
}

// -----

// Effective kernel = (2 - 1) * 2 + 1 = 3. The raw extent 2 is corrected to 1.
func.func @valid_ceil_trailing_window_dilation(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x3xf32, 1>,
    %output: memref<1x1x1xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x1x3xf32, 1>)
    outs(%output : memref<1x1x1xf32, 1>)
    {pool_mode = 1, kernel_shape = [2], strides = [3], pads = [0, 1],
     dilations = [2], ceil_mode = 1, storage_order = 0}
  return
}

// -----

// A zero input with kernel 1 first produces a raw ceil extent of 1; the
// trailing-window correction brings it to zero without underflowing to -1.
func.func @valid_ceil_zero_extent(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x0xf32, 1>,
    %output: memref<1x1x0xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x1x0xf32, 1>)
    outs(%output : memref<1x1x0xf32, 1>)
    {pool_mode = 0, kernel_shape = [1], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

// Keep the final-window multiplication in signed wide arithmetic: both the
// stride and `(output - 1) * stride` reach INT64_MAX before correction.
func.func @valid_ceil_i64_boundary(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x9223372036854775807xf32, 1>,
    %output: memref<1x1x1xf32, 1>) {
  hip.pool(%ctx)
    valid(%valid)
    ins(%input : memref<1x1x9223372036854775807xf32, 1>)
    outs(%output : memref<1x1x1xf32, 1>)
    {pool_mode = 1, kernel_shape = [1], strides = [9223372036854775807],
     pads = [0, 0], dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

func.func @wrong_uncorrected_ceil_shape(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x4xf32, 1>,
    %output: memref<1x3x3xf32, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 2}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x4xf32, 1>)
    outs(%output : memref<1x3x3xf32, 1>)
    {pool_mode = 1, kernel_shape = [3], strides = [2], pads = [0, 2],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}

// -----

func.func @wrong_values_shape(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x8x8xf32, 1>,
    %output: memref<1x3x5x4xf32, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 4}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8x8xf32, 1>)
    outs(%output : memref<1x3x5x4xf32, 1>)
    {pool_mode = 1, kernel_shape = [2, 2], strides = [2, 2],
     pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0,
     storage_order = 0}
  return
}

// -----

func.func @wrong_indices_shape(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x8x8xf32, 1>,
    %values: memref<1x3x4x4xf32, 1>,
    %indices: memref<1x3x4x5xi64, 1>) {
  // expected-error @+1 {{dim 3 of result mismatch: expected 4}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8x8xf32, 1>)
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
    %valid: i1,
    %input: memref<1x3x8xf32, 1>,
    %values: memref<1x3x4xf32, 1>,
    %indices: memref<1x3x4xi64, 1>) {
  // expected-error @+1 {{only max pooling may produce an indices output}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8xf32, 1>)
    outs(%values, %indices : memref<1x3x4xf32, 1>,
                             memref<1x3x4xi64, 1>)
    {pool_mode = 0, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @wrong_indices_type(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x8xf32, 1>,
    %values: memref<1x3x4xf32, 1>,
    %indices: memref<1x3x4xi32, 1>) {
  // expected-error @+1 {{indices output must have i64 element type}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8xf32, 1>)
    outs(%values, %indices : memref<1x3x4xf32, 1>,
                             memref<1x3x4xi32, 1>)
    {pool_mode = 1, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @invalid_pool_mode(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x8xf32, 1>,
    %output: memref<1x3x4xf32, 1>) {
  // expected-error @+1 {{pool_mode must be 0}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8xf32, 1>)
    outs(%output : memref<1x3x4xf32, 1>)
    {pool_mode = 3, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @wrong_values_type(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x3x8xf32, 1>,
    %output: memref<1x3x4xf16, 1>) {
  // expected-error @+1 {{input and values output must have the same floating-point type}}
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x3x8xf32, 1>)
    outs(%output : memref<1x3x4xf16, 1>)
    {pool_mode = 1, kernel_shape = [2], strides = [2], pads = [0, 0],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @wide_window_product(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x?xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x1x?xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {pool_mode = 0, kernel_shape = [9223372036854775807], strides = [1],
     pads = [0, 0], dilations = [9223372036854775807], ceil_mode = 0,
     storage_order = 0}
  return
}

// -----

func.func @window_offset_i64_boundary(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x?xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  hip.pool(%ctx) valid(%valid) ins(%input : memref<1x1x?xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {pool_mode = 0, kernel_shape = [9223372036854775807], strides = [1],
     pads = [9223372036854775807, 9223372036854775807],
     dilations = [1], ceil_mode = 0, storage_order = 0}
  return
}

// -----

func.func @static_output_out_of_i64_range(
    %ctx: !hip.context,
    %valid: i1,
    %input: memref<1x1x9223372036854775807xf32, 1>,
    %output: memref<1x1x?xf32, 1>) {
  // expected-error @+1 {{pool inferred an out-of-range output extent at spatial axis 0}}
  hip.pool(%ctx) valid(%valid)
    ins(%input : memref<1x1x9223372036854775807xf32, 1>)
    outs(%output : memref<1x1x?xf32, 1>)
    {pool_mode = 1, kernel_shape = [1], strides = [1],
     pads = [9223372036854775807, 9223372036854775807],
     dilations = [1], ceil_mode = 1, storage_order = 0}
  return
}
