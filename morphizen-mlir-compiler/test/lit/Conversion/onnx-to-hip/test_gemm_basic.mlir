// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX GEMM (General Matrix Multiplication) is correctly lowered to
// hip.gemm operation.
//
// This test validates:
// - GEMM operation lowering (onnx.Gemm → hip.gemm)
// - Matrix multiplication with bias: Y = alpha * A * B + beta * C
// - Attribute preservation (alpha, beta, transA, transB)
// - 2D matrix operand handling
//
// Input: GEMM with standard parameters (alpha=1.0, beta=1.0, no transpose)
//        Matrix sizes: A[128x256] * B[256x512] + C[128x512] → Y[128x512]
// Expected: hip.gemm operation with identical attributes
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @gemm_test(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>, %C: tensor<128x512xf32>) -> tensor<128x512xf32> {
    // After conversion: context added, tensors→memrefs, output arg added, return→i32
    // CHECK-LABEL: func.func @gemm_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: memref<128x256xf32, 1>, %[[B:.*]]: memref<256x512xf32, 1>, %[[C:.*]]: memref<128x512xf32, 1>, %[[OUTPUT_ARG:.*]]: memref<128x512xf32, 1>) -> i32

    // ONNX GEMM: Y = alpha * A * B + beta * C
    %output = "onnx.Gemm"(%A, %B, %C) {
      alpha = 1.0 : f32,
      beta = 1.0 : f32,
      transA = 0 : si64,
      transB = 0 : si64
    } : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>

    // After conversion: allocate temp, call hip.gemm, copy to output arg
    // CHECK: %[[ALLOC:.*]] = hip.alloc(%[[CTX]]) : memref<128x512xf32, 1>
    // CHECK-NEXT: hip.gemm(%[[CTX]], %[[A]], %[[B]], %[[C]], %[[ALLOC]])
    // CHECK-SAME: {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64}
    // CHECK: hip.copy(%[[CTX]], %[[ALLOC]], %[[OUTPUT_ARG]])
    // CHECK: arith.constant 0 : i32

    return %output : tensor<128x512xf32>
  }
}
