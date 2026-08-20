// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s --convert-hip-to-llvm 2>&1 | FileCheck %s

// CHECK-DAG: 'hip.reduce_max' op unsupported reduction element type 'f32'; supported types: f16, i32, i64
// CHECK-DAG: 'hip.reduce_min' op unsupported reduction element type 'bf16'; supported types: f16, i32, i64
// CHECK-DAG: 'hip.reduce_prod' op unsupported reduction element type 'f64'; supported types: f16, i32, i64

module {
  memref.global "private" constant @axis1 : memref<1xi64> = dense<[1]>

  func.func @reject_reduce_max(%ctx: !hip.context, %data: memref<2x3xf32>,
                               %output: memref<2xf32>) {
    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_max(%ctx)
        ins(%data, %axes : memref<2x3xf32>, memref<1xi64>)
        outs(%output : memref<2xf32>)
        {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
         normalized_axes = array<i64: 1>}
    return
  }

  func.func @reject_reduce_min(%ctx: !hip.context, %data: memref<2x3xbf16>,
                               %output: memref<2xbf16>) {
    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_min(%ctx)
        ins(%data, %axes : memref<2x3xbf16>, memref<1xi64>)
        outs(%output : memref<2xbf16>)
        {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
         normalized_axes = array<i64: 1>}
    return
  }

  func.func @reject_reduce_prod(%ctx: !hip.context, %data: memref<2x3xf64>,
                                %output: memref<2xf64>) {
    %axes = memref.get_global @axis1 : memref<1xi64>
    hip.reduce_prod(%ctx)
        ins(%data, %axes : memref<2x3xf64>, memref<1xi64>)
        outs(%output : memref<2xf64>)
        {keepdims = 0 : i64, noop_with_empty_axes = 0 : i64,
         normalized_axes = array<i64: 1>}
    return
  }
}
