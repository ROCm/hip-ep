// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP gemm operation is correctly lowered to DynamicDispatch backend
// when --use-dynamic-dispatch option is enabled.
//
// This test validates:
// - hip.gemm → llvm.call @wrap_dd_matmul (NPU/IPU backend)
// - Function signature matches DynamicDispatch C ABI
// - 14-param signature: state, op_state_slot, input_a, input_b, bias, output,
//                        M, N, K, alpha, beta, transA, transB, data_type
// - Optional bias (input C): present → pointer, absent → null pointer
// - Data type passed as i64 constant
// - alpha/beta passed as f64 (not f32 like GPU backend)
//
// Expected: wrap_dd_matmul(state, op_state_slot, A_ptr, B_ptr, bias_ptr, output_ptr,
//             M, N, K, alpha, beta, transA, transB, data_type)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm="use-dynamic-dispatch=true" | FileCheck %s

module {
  // Test 1: f16 GEMM with bias, default attributes
  // A=[128,256], B=[256,512], C=[512] (bias), Y=[128,512]
  // M=128, K=256, N=512, typeCode=0 (f16), alpha=1.0, beta=0.0
  func.func @dd_gemm_f16_with_bias(
      %ctx: !hip.context,
      %a: memref<128x256xf16, 1>,
      %b: memref<256x512xf16, 1>,
      %c: memref<512xf16, 1>,
      %y: memref<128x512xf16, 1>) {
    // CHECK-DAG: llvm.func @wrap_dd_matmul(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32
    // CHECK-LABEL: llvm.func @dd_gemm_f16_with_bias
    // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

    hip.gemm(%ctx) ins(%a, %b, %c : memref<128x256xf16, 1>, memref<256x512xf16, 1>, memref<512xf16, 1>)
                   outs(%y : memref<128x512xf16, 1>)

    return
  }

  // Test 2: f32 GEMM without bias
  // A=[64,128], B=[128,256], Y=[64,256]
  // M=64, K=128, N=256, typeCode=1 (f32), no bias (null pointer)
  func.func @dd_gemm_f32_no_bias(
      %ctx: !hip.context,
      %a: memref<64x128xf32, 1>,
      %b: memref<128x256xf32, 1>,
      %y: memref<64x256xf32, 1>) {
    // CHECK-LABEL: llvm.func @dd_gemm_f32_no_bias

    hip.gemm(%ctx) ins(%a, %b : memref<64x128xf32, 1>, memref<128x256xf32, 1>)
                   outs(%y : memref<64x256xf32, 1>)

    // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

    return
  }

  // Test 3: transB flag
  // A=[32,64], B=[128,64]^T (transB=1), Y=[32,128]
  // M=32, K=64, N=128, transB=1
  func.func @dd_gemm_transB(
      %ctx: !hip.context,
      %a: memref<32x64xf16, 1>,
      %b: memref<128x64xf16, 1>,
      %y: memref<32x128xf16, 1>) {
    // CHECK-LABEL: llvm.func @dd_gemm_transB

    hip.gemm(%ctx) ins(%a, %b : memref<32x64xf16, 1>, memref<128x64xf16, 1>)
                   outs(%y : memref<32x128xf16, 1>)
                   {transB = 1 : i64}

    // Verify transB=1 is passed as i64 constant
    // CHECK: llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

    return
  }

  // Test 4: Custom alpha/beta
  // A=[16,32], B=[32,64], C=[64], Y=[16,64]
  // alpha=2.0, beta=0.5
  func.func @dd_gemm_custom_alpha_beta(
      %ctx: !hip.context,
      %a: memref<16x32xf32, 1>,
      %b: memref<32x64xf32, 1>,
      %c: memref<64xf32, 1>,
      %y: memref<16x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @dd_gemm_custom_alpha_beta

    hip.gemm(%ctx) ins(%a, %b, %c : memref<16x32xf32, 1>, memref<32x64xf32, 1>, memref<64xf32, 1>)
                   outs(%y : memref<16x64xf32, 1>)
                   {alpha = 2.0 : f32, beta = 0.5 : f32}

    // Verify alpha=2.0, beta=0.5 passed as f64
    // CHECK-DAG: llvm.mlir.constant(2.000000e+00 : f64) : f64
    // CHECK-DAG: llvm.mlir.constant(5.000000e-01 : f64) : f64
    // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

    return
  }

  // Test 5: bfloat16 data type
  // A=[8,16], B=[16,32], Y=[8,32]
  // data_type should be HIPDNN_EP_DATATYPE_BFLOAT16
  func.func @dd_gemm_bf16(
      %ctx: !hip.context,
      %a: memref<8x16xbf16, 1>,
      %b: memref<16x32xbf16, 1>,
      %y: memref<8x32xbf16, 1>) {
    // CHECK-LABEL: llvm.func @dd_gemm_bf16

    hip.gemm(%ctx) ins(%a, %b : memref<8x16xbf16, 1>, memref<16x32xbf16, 1>)
                   outs(%y : memref<8x32xbf16, 1>)

    // Verify bfloat16 data type constant (HIPDNN_EP_DATATYPE_BFLOAT16 = 2)
    // CHECK: llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

    return
  }
}
