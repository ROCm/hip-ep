// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @batch_axis(%ctx: !hip.context, %past: memref<4x16x8xf16, 1>,
    %update: memref<1x16x8xf16, 1>, %out: memref<4x16x8xf16, 1>) {
  // expected-error @+1 {{axis must be in [1, rank)}}
  hip.tensor_scatter(%ctx) ins(%past, %update : memref<4x16x8xf16, 1>,
                               memref<1x16x8xf16, 1>)
      outs(%out : memref<4x16x8xf16, 1>) {axis = 0 : i64}
  return
}

// -----

func.func @update_row_width(%ctx: !hip.context, %past: memref<1x16x8xf16, 1>,
    %update: memref<1x2x4xf16, 1>, %out: memref<1x16x8xf16, 1>) {
  // expected-error @+1 {{update and output must match past outside axis, output everywhere}}
  hip.tensor_scatter(%ctx) ins(%past, %update : memref<1x16x8xf16, 1>,
                               memref<1x2x4xf16, 1>)
      outs(%out : memref<1x16x8xf16, 1>) {axis = 1 : i64}
  return
}

// -----

func.func @write_indices_type(%ctx: !hip.context, %past: memref<1x16x8xf16, 1>,
    %update: memref<1x2x8xf16, 1>, %wi: memref<1xi32, 1>,
    %out: memref<1x16x8xf16, 1>) {
  // expected-error @+1 {{write_indices must be [B] int64}}
  hip.tensor_scatter(%ctx) ins(%past, %update : memref<1x16x8xf16, 1>,
                               memref<1x2x8xf16, 1>)
      write_indices(%wi : memref<1xi32, 1>)
      outs(%out : memref<1x16x8xf16, 1>) {axis = 1 : i64}
  return
}
