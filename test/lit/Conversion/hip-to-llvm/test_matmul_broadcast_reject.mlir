// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s -split-input-file -verify-diagnostics

// A [2, 1, M, K] x B [1, 3, K, N] broadcasts each operand over a different
// batch dim. A strided-batched GEMM has one stride per operand, so no stride
// pair reads the right matrices; the lowering rejects it instead of running
// 2 batches over the 6-batch output.

module {
  func.func @matmul_cross_broadcast(%ctx: !hip.context,
                                    %A: memref<2x1x16x64xf16, 1>,
                                    %B: memref<1x3x64x32xf16, 1>,
                                    %output: memref<2x3x16x32xf16, 1>) {
    // expected-error @+2 {{batch broadcast between two multi-matrix operands}}
    // expected-error @+1 {{failed to legalize operation 'hip.matmul'}}
    hip.matmul(%ctx)
        ins(%A, %B : memref<2x1x16x64xf16, 1>, memref<1x3x64x32xf16, 1>)
        outs(%output : memref<2x3x16x32xf16, 1>)
    return
  }
}
