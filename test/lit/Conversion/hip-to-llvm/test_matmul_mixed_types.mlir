// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file --verify-diagnostics %s

func.func @mismatched_b(%ctx: !hip.context, %a: memref<2x2xbf16, 1>,
                        %b: memref<2x2xf16, 1>, %out: memref<2x2xbf16, 1>) {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul'}}
  hip.matmul(%ctx) ins(%a, %b : memref<2x2xbf16, 1>, memref<2x2xf16, 1>)
      outs(%out : memref<2x2xbf16, 1>)
  return
}

// -----

func.func @mismatched_output(%ctx: !hip.context, %a: memref<2x2xf32, 1>,
                             %b: memref<2x2xf32, 1>, %out: memref<2x2xf16, 1>) {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul'}}
  hip.matmul(%ctx) ins(%a, %b : memref<2x2xf32, 1>, memref<2x2xf32, 1>)
      outs(%out : memref<2x2xf16, 1>)
  return
}

// -----

func.func @mismatched_i32(%ctx: !hip.context, %a: memref<2x2xi32, 1>,
                          %b: memref<2x2xf32, 1>, %out: memref<2x2xi32, 1>) {
  // expected-error @+1 {{failed to legalize operation 'hip.matmul'}}
  hip.matmul(%ctx) ins(%a, %b : memref<2x2xi32, 1>, memref<2x2xf32, 1>)
      outs(%out : memref<2x2xi32, 1>)
  return
}
