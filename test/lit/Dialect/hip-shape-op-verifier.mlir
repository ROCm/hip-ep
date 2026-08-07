// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @gather_valid(%ctx: !hip.context,
                        %data: memref<2x3x4xf32, 1>,
                        %indices: memref<5xi64, 1>,
                        %out: memref<2x5x4xf32, 1>) {
  hip.gather(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>, memref<5xi64, 1>)
    outs(%out : memref<2x5x4xf32, 1>)
    {axis = 1 : i64}
  return
}

// -----

func.func @gather_wrong_output(%ctx: !hip.context,
                               %data: memref<2x3x4xf32, 1>,
                               %indices: memref<5xi64, 1>,
                               %out: memref<2x6x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 5}}
  hip.gather(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>, memref<5xi64, 1>)
    outs(%out : memref<2x6x4xf32, 1>)
    {axis = -2 : i64}
  return
}

// -----

func.func @gather_nd_valid(%ctx: !hip.context,
                           %data: memref<2x3x4xf32, 1>,
                           %indices: memref<2x5x1xi64, 1>,
                           %out: memref<2x5x4xf32, 1>) {
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>,
         memref<2x5x1xi64, 1>)
    outs(%out : memref<2x5x4xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @gather_nd_dynamic_tuple_fallback(
    %ctx: !hip.context,
    %data: memref<?x?x?xf32, 1>,
    %indices: memref<?x?xi64, 1>,
    %out: memref<?x?xf32, 1>) {
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<?x?x?xf32, 1>, memref<?x?xi64, 1>)
    outs(%out : memref<?x?xf32, 1>)
    {batch_dims = 0 : i64}
  return
}

// -----

func.func @gather_nd_tuple_width_addition_overflow(
    %ctx: !hip.context,
    %data: memref<1x1xf32, 1>,
    %indices: memref<1x9223372036854775807xi64, 1>,
    %out: memref<?xf32, 1>) {
  // expected-error @+1 {{indices tuple width and batch_dims are incompatible with data rank}}
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<1x1xf32, 1>,
         memref<1x9223372036854775807xi64, 1>)
    outs(%out : memref<?xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @gather_nd_bad_batch_prefix(
    %ctx: !hip.context,
    %data: memref<2x3x4xf32, 1>,
    %indices: memref<7x5x1xi64, 1>,
    %out: memref<7x5x4xf32, 1>) {
  // expected-error @+1 {{data and indices batch dimensions must match}}
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>,
         memref<7x5x1xi64, 1>)
    outs(%out : memref<7x5x4xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @transpose_valid(%ctx: !hip.context,
                           %input: memref<2x3x4xf16, 1>,
                           %out: memref<4x2x3xf16, 1>) {
  hip.transpose(%ctx)
    ins(%input : memref<2x3x4xf16, 1>)
    outs(%out : memref<4x2x3xf16, 1>)
    {perm = [2, 0, 1]}
  return
}

// -----

func.func @transpose_wrong_output(%ctx: !hip.context,
                                  %input: memref<2x3x4xf16, 1>,
                                  %out: memref<4x3x2xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 2}}
  hip.transpose(%ctx)
    ins(%input : memref<2x3x4xf16, 1>)
    outs(%out : memref<4x3x2xf16, 1>)
    {perm = [2, 0, 1]}
  return
}

// -----

func.func @matmul_nbits_valid(
    %ctx: !hip.context,
    %a: memref<2x3x64xf16, 1>,
    %b: memref<32x2x16xui8, 1>,
    %scales: memref<32x2xf16, 1>,
    %out: memref<2x3x32xf16, 1>) {
  hip.matmul_nbits(%ctx)
    ins(%a, %b, %scales : memref<2x3x64xf16, 1>,
        memref<32x2x16xui8, 1>, memref<32x2xf16, 1>)
    outs(%out : memref<2x3x32xf16, 1>)
    {K = 64 : i64, N = 32 : i64, bits = 4 : i64,
     block_size = 32 : i64, accuracy_level = 0 : i64,
     zp_elem_size = 2 : i64}
  return
}

// -----

func.func @matmul_nbits_wrong_output(
    %ctx: !hip.context,
    %a: memref<2x3x64xf16, 1>,
    %b: memref<32x2x16xui8, 1>,
    %scales: memref<32x2xf16, 1>,
    %out: memref<2x3x31xf16, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 32}}
  hip.matmul_nbits(%ctx)
    ins(%a, %b, %scales : memref<2x3x64xf16, 1>,
        memref<32x2x16xui8, 1>, memref<32x2xf16, 1>)
    outs(%out : memref<2x3x31xf16, 1>)
    {K = 64 : i64, N = 32 : i64, bits = 4 : i64,
     block_size = 32 : i64, accuracy_level = 0 : i64,
     zp_elem_size = 2 : i64}
  return
}

// -----

func.func @size_valid(%ctx: !hip.context,
                      %input: memref<?x4xf32, 1>,
                      %out: memref<i64, 1>) {
  hip.size(%ctx)
    ins(%input : memref<?x4xf32, 1>)
    outs(%out : memref<i64, 1>)
  return
}

// -----

func.func @size_wrong_rank(%ctx: !hip.context,
                           %input: memref<?x4xf32, 1>,
                           %out: memref<1xi64, 1>) {
  // expected-error @+1 {{output must be a rank-zero i64 tensor or memref}}
  hip.size(%ctx)
    ins(%input : memref<?x4xf32, 1>)
    outs(%out : memref<1xi64, 1>)
  return
}
