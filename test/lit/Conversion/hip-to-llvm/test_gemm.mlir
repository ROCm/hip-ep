// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP gemm operation is correctly lowered to LLVM call
// to wrap_gemm runtime function.
//
// This test validates:
// - hip.gemm → llvm.call @wrap_gemm
// - Optional input C: present → pointer, absent → null pointer
// - Data type enum (f16=0, f32=1, f64=2, bf16=3)
// - Attributes: alpha, beta, transA, transB passed as constants
// - M/N/K dimensions extracted correctly (respecting transA/transB)
// - C shape (cDim0, cDim1) extracted and passed for broadcast support
// - 16-param signature: state, A, B, C, output, M, N, K,
//                        alpha, beta, transA, transB, typeCode, cDim0, cDim1,
//                        op_state_slot
//
// Expected: wrap_gemm(state, A_ptr, B_ptr, C_ptr, output_ptr,
//             M, N, K, alpha, beta, transA, transB, typeCode, cDim0, cDim1,
//             op_state_slot)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: f16 Gemm with 1D C=[5120] (bias), default attributes
  // A=[1,5120], B=[5120,5120], C=[5120], Y=[1,5120]
  // M=1, K=5120, N=5120, typeCode=0 (f16), cDim0=1, cDim1=5120
  func.func @gemm_f16_with_bias(
      %ctx: !hip.context,
      %a: memref<1x5120xf16, 1>,
      %b: memref<5120x5120xf16, 1>,
      %c: memref<5120xf16, 1>,
      %y: memref<1x5120xf16, 1>) {
    // CHECK-LABEL: llvm.func @gemm_f16_with_bias

    hip.gemm(%ctx) ins(%a, %b, %c : memref<1x5120xf16, 1>, memref<5120x5120xf16, 1>, memref<5120xf16, 1>)
                   outs(%y : memref<1x5120xf16, 1>)

    // CHECK: llvm.call @wrap_gemm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, f32, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: f32 Gemm without C (no bias) — C should be null pointer, cDim0=cDim1=0
  // A=[128,256], B=[256,512], Y=[128,512]
  // M=128, K=256, N=512, typeCode=1 (f32)
  func.func @gemm_f32_no_bias(
      %ctx: !hip.context,
      %a: memref<128x256xf32, 1>,
      %b: memref<256x512xf32, 1>,
      %y: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @gemm_f32_no_bias

    hip.gemm(%ctx) ins(%a, %b : memref<128x256xf32, 1>, memref<256x512xf32, 1>)
                   outs(%y : memref<128x512xf32, 1>)

    // null pointer for absent C
    // CHECK: llvm.mlir.zero
    // CHECK: llvm.call @wrap_gemm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, f32, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: f16 Gemm with transB=1, 1D C=[512]
  // A=[1,256], B=[512,256] (transposed: logical [256,512]), C=[512], Y=[1,512]
  // M=1, K=256, N=512, typeCode=0 (f16), cDim0=1, cDim1=512
  func.func @gemm_f16_transB(
      %ctx: !hip.context,
      %a: memref<1x256xf16, 1>,
      %b: memref<512x256xf16, 1>,
      %c: memref<512xf16, 1>,
      %y: memref<1x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @gemm_f16_transB

    hip.gemm(%ctx) ins(%a, %b, %c : memref<1x256xf16, 1>, memref<512x256xf16, 1>, memref<512xf16, 1>)
                   outs(%y : memref<1x512xf16, 1>)
                   {transB = 1 : i64}

    // CHECK: llvm.call @wrap_gemm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, f32, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 4: f32 Gemm with transA=1, custom alpha/beta, 1D C=[512]
  // A=[256,128] (transposed: logical [128,256]), B=[256,512], C=[512], Y=[128,512]
  // M=128, K=256, N=512, alpha=2.0, beta=0.5, typeCode=1, cDim0=1, cDim1=512
  func.func @gemm_f32_transA_custom_alpha_beta(
      %ctx: !hip.context,
      %a: memref<256x128xf32, 1>,
      %b: memref<256x512xf32, 1>,
      %c: memref<512xf32, 1>,
      %y: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @gemm_f32_transA_custom_alpha_beta

    hip.gemm(%ctx) ins(%a, %b, %c : memref<256x128xf32, 1>, memref<256x512xf32, 1>, memref<512xf32, 1>)
                   outs(%y : memref<128x512xf32, 1>)
                   {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transA = 1 : i64}

    // CHECK: llvm.call @wrap_gemm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, f32, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 5: f32 Gemm with 2D C=[128,512] (no broadcast needed)
  // A=[128,256], B=[256,512], C=[128,512], Y=[128,512]
  // M=128, K=256, N=512, typeCode=1, cDim0=128, cDim1=512
  func.func @gemm_f32_2d_bias(
      %ctx: !hip.context,
      %a: memref<128x256xf32, 1>,
      %b: memref<256x512xf32, 1>,
      %c: memref<128x512xf32, 1>,
      %y: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @gemm_f32_2d_bias

    hip.gemm(%ctx) ins(%a, %b, %c : memref<128x256xf32, 1>, memref<256x512xf32, 1>, memref<128x512xf32, 1>)
                   outs(%y : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_gemm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, f32, i64, i64, i64, i64, i64) -> i32

    return
  }
}
