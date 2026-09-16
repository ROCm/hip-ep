// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-use-output-allocator

func.func @unbufferized_shape(
    %ctx: !hipsr.context, %M: index, %N: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %shape = shape.from_extents %M, %N : index, index
  %out = memref.alloc(%M, %N) : memref<?x?xf16, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %out
    : !shape.shape, memref<?x?xf16, #hipsr.mem<device>>
  // expected-error@+1 {{graph output 0 has dynamic dims but no preserved external shape memref}}
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}

// -----

func.func @missing_shape(
    %ctx: !hipsr.context, %M: index, %N: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %out = memref.alloc(%M, %N) : memref<?x?xf16, #hipsr.mem<device>>
  // expected-error@+1 {{graph output 0 has dynamic dims but no preserved external shape memref}}
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}
