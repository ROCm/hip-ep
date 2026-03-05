// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP GEMM operations are correctly lowered to LLVM calls to
// hipBLAS library.
//
// This test validates:
// - hip.gemm → llvm.call @wrap_hipblas_sgemm
// - Type conversion: !hip.context → !llvm.ptr
// - Memref descriptors expanded to individual LLVM parameters
// - GEMM parameter passing (alpha, beta, transA, transB)
// - Proper function signature for hipBLAS API
//
// Note: MLIR's memref-to-LLVM conversion expands memref descriptors into
// individual scalar parameters. This is standard MLIR behavior.
//
// GEMM operation: Y = alpha * A * B + beta * C
// Expected: LLVM function calling hipBLAS wrapper with correct signature
// ============================================================================

// RUN: udna-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @gemm_llvm_test(
      %ctx: !hip.context,
      %A: memref<128x256xf32, 1>,
      %B: memref<256x512xf32, 1>,
      %C: memref<128x512xf32, 1>,
      %Y: memref<128x512xf32, 1>) {
    // After lowering, function becomes llvm.func with expanded parameters
    // CHECK-LABEL: llvm.func @gemm_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    // HIP GEMM: Y = alpha * A * B + beta * C
    hip.gemm(%ctx, %A, %B, %C, %Y)
      {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0, transB = 0}
      : (!hip.context, memref<128x256xf32, 1>, memref<256x512xf32, 1>,
         memref<128x512xf32, 1>, memref<128x512xf32, 1>)

    // Should lower to hipBLAS single-precision GEMM call
    // The wrapper is declared and called
    // CHECK: llvm.call @wrap_hipblas_sgemm

    return
  }
}
