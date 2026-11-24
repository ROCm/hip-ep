// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Runtime axes with dynamic shapes that fit more than one reduced axis:
// [?, ?, ?] -> [?, ?] could reduce any of the three. inner_size = 1 would be
// right only for the last one, so the lowering is rejected instead of
// guessing.

// RUN: not hip-mlir-opt --convert-hip-to-llvm %s 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'hip.reduce_mean'
func.func @reduce_mean_ambiguous_axis(%ctx: !hip.context,
                                      %input: memref<?x?x?xf16, 1>,
                                      %axes: memref<1xi64, 1>,
                                      %output: memref<?x?xf16, 1>) {
  hip.reduce_mean(%ctx) ins(%input, %axes : memref<?x?x?xf16, 1>, memref<1xi64, 1>)
                        outs(%output : memref<?x?xf16, 1>)
                        {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64}
  return
}
