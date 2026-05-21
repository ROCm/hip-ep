// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_matmul_lowering(%ctx: !hip.context,
                                   %A: memref<1x128x4096xf16, 1>,
                                   %B: memref<4096x1024xf16, 1>,
                                   %output: memref<1x128x1024xf16, 1>) {
    hip.matmul(%ctx)
        ins(%A, %B : memref<1x128x4096xf16, 1>, memref<4096x1024xf16, 1>)
        outs(%output : memref<1x128x1024xf16, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_matmul_lowering
// CHECK: llvm.call @wrap_hipblasLtMatmul({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32
// Verify 10 parameters:
// - 4 pointers: state, A, B, output
// - 6 i64: M=128, N=1024, K=4096, batch_count=1, elem_size=2, b_batched=0
//   (B is rank-2 [K, N] = broadcast weight → b_batched = 0)
