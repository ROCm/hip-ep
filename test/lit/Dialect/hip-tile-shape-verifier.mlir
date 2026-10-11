// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid(%ctx: !hip.context,
                 %input: memref<2x3xf32, 1>,
                 %repeats: memref<2xi64, 1>,
                 %output: memref<4x9xf32, 1>) {
  hip.tile(%ctx)
    ins(%input, %repeats : memref<2x3xf32, 1>, memref<2xi64, 1>)
    outs(%output : memref<4x9xf32, 1>)
    {static_repeats = array<i64: 2, 3>}
  return
}

// -----

func.func @wrong_output(%ctx: !hip.context,
                        %input: memref<2x3xf32, 1>,
                        %repeats: memref<2xi64, 1>,
                        %output: memref<2x3xf32, 1>) {
  // expected-error @+1 {{dim 0 of result mismatch: expected 4}}
  hip.tile(%ctx)
    ins(%input, %repeats : memref<2x3xf32, 1>, memref<2xi64, 1>)
    outs(%output : memref<2x3xf32, 1>)
    {static_repeats = array<i64: 2, 3>}
  return
}

// -----

func.func @negative_repeat(%ctx: !hip.context,
                           %input: memref<2x3xf32, 1>,
                           %repeats: memref<2xi64, 1>,
                           %output: memref<0x3xf32, 1>) {
  // expected-error @+1 {{static_repeats must match input rank and be non-negative}}
  hip.tile(%ctx)
    ins(%input, %repeats : memref<2x3xf32, 1>, memref<2xi64, 1>)
    outs(%output : memref<0x3xf32, 1>)
    {static_repeats = array<i64: -1, 1>}
  return
}

// -----

func.func @bad_repeats_type(%ctx: !hip.context,
                            %input: memref<2x3xf32, 1>,
                            %repeats: memref<2xi32, 1>,
                            %output: memref<4x9xf32, 1>) {
  // expected-error @+1 {{repeats must be static-length rank-1 i64 matching input rank}}
  hip.tile(%ctx)
    ins(%input, %repeats : memref<2x3xf32, 1>, memref<2xi32, 1>)
    outs(%output : memref<4x9xf32, 1>)
    {static_repeats = array<i64: 2, 3>}
  return
}

// -----

func.func @repeat_product_overflow(
    %ctx: !hip.context,
    %input: memref<2xf32, 1>,
    %repeats: memref<1xi64, 1>,
    %output: memref<?xf32, 1>) {
  // expected-error @+1 {{static_repeats must match input rank and be non-negative}}
  hip.tile(%ctx)
    ins(%input, %repeats : memref<2xf32, 1>, memref<1xi64, 1>)
    outs(%output : memref<?xf32, 1>)
    {static_repeats = array<i64: 9223372036854775807>}
  return
}

// -----

func.func @repeat_product_i64_boundary(
    %ctx: !hip.context,
    %input: memref<1xf32, 1>,
    %repeats: memref<1xi64, 1>,
    %output: memref<9223372036854775807xf32, 1>) {
  hip.tile(%ctx)
    ins(%input, %repeats : memref<1xf32, 1>, memref<1xi64, 1>)
    outs(%output : memref<9223372036854775807xf32, 1>)
    {static_repeats = array<i64: 9223372036854775807>}
  return
}
