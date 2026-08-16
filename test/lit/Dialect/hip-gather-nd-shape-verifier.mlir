// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @valid_static_tuple_width(
    %ctx: !hip.context,
    %data: memref<2x3x4xf32, 1>,
    %indices: memref<2x2x1xi64, 1>,
    %output: memref<2x2x4xf32, 1>) {
  hip.gather_nd(%ctx)
      ins(%data, %indices : memref<2x3x4xf32, 1>, memref<2x2x1xi64, 1>)
      outs(%output : memref<2x2x4xf32, 1>)
      {batch_dims = 1 : i64}
  return
}

// -----

// A dynamic tuple width cannot be reified from the operands, but remains valid
// with the DPS destination as the shape authority.
func.func @valid_dynamic_tuple_width(
    %ctx: !hip.context,
    %data: memref<2x2xf32, 1>,
    %indices: memref<2x?xi64, 1>,
    %output: memref<2xf32, 1>) {
  hip.gather_nd(%ctx)
      ins(%data, %indices : memref<2x2xf32, 1>, memref<2x?xi64, 1>)
      outs(%output : memref<2xf32, 1>)
  return
}

// -----

func.func @reject_i32_indices(
    %ctx: !hip.context,
    %data: memref<2x2xf32, 1>,
    %indices: memref<2x2xi32, 1>,
    %output: memref<2xf32, 1>) {
  // expected-error @+1 {{'hip.gather_nd' op indices element type must be i64 because the runtime ABI reads int64 indices}}
  hip.gather_nd(%ctx)
      ins(%data, %indices : memref<2x2xf32, 1>, memref<2x2xi32, 1>)
      outs(%output : memref<2xf32, 1>)
  return
}
