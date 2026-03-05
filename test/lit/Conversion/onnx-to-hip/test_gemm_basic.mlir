// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX GEMM (General Matrix Multiplication) is correctly lowered to
// hip.gemm operation in tensor-first mode.
//
// This test validates:
// - GEMM operation lowering (onnx.Gemm → hip.gemm)
// - Matrix multiplication with bias: Y = alpha * A * B + beta * C
// - Attribute preservation (alpha, beta, transA, transB)
// - 2D matrix operand handling
// - Tensor-first DPS: tensor.empty() used as output init
//
// Input: GEMM with standard parameters (alpha=1.0, beta=1.0, no transpose)
//        Matrix sizes: A[128x256] * B[256x512] + C[128x512] → Y[128x512]
// Expected: hip.gemm in tensor mode with identical attributes
// ============================================================================

// RUN: udna-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @gemm_test(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>, %C: tensor<128x512xf32>) -> tensor<128x512xf32> {
    // After conversion: context prepended, tensors remain tensors, tensor return
    // CHECK-LABEL: func.func @gemm_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<128x256xf32>, %[[B:.*]]: tensor<256x512xf32>, %[[C:.*]]: tensor<128x512xf32>) -> tensor<128x512xf32>

    // ONNX GEMM: Y = alpha * A * B + beta * C
    %output = "onnx.Gemm"(%A, %B, %C) {
      alpha = 1.0 : f32,
      beta = 1.0 : f32,
      transA = 0 : si64,
      transB = 0 : si64
    } : (tensor<128x256xf32>, tensor<256x512xf32>, tensor<128x512xf32>) -> tensor<128x512xf32>

    // After conversion: tensor.empty() for init, hip.gemm in tensor mode
    // CHECK: tensor.empty() : tensor<128x512xf32>
    // CHECK: hip.gemm(%[[CTX]], %[[A]], %[[B]], %[[C]], {{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : i64, transB = 0 : i64}
    // CHECK-NOT: hip.alloc
    // CHECK-NOT: hip.copy

    return %output : tensor<128x512xf32>
  }
}
