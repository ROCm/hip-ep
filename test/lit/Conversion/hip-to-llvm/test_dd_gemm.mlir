// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// With --use-dynamic-dispatch, hip.gemm / hip.matmul lower to wrap_dd_matmul
// (NPU) only for the shapes DynamicDispatch has kernels for
// (include/hip/dynamic_dispatch_support.h): f16, rank 2, no transpose, no
// bias, alpha 1, and [M, K] x [K, N] in {[1,1024]x[1024,192],
// [1,1024]x[1024,384]}. Everything else lowers to the GPU path.
//
// wrap_dd_matmul signature: state, op_state_slot, input_a, input_b, bias,
// output, M, N, K, alpha, beta, transA, transB, data_type.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm="use-dynamic-dispatch=true" | FileCheck %s

module {
  // CHECK-DAG: llvm.func @wrap_dd_matmul(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32

  // CHECK-LABEL: llvm.func @dd_gemm_n384
  // CHECK: llvm.call @wrap_dd_matmul({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f64, f64, i64, i64, i64) -> i32
  func.func @dd_gemm_n384(%ctx: !hip.context,
                          %a: memref<1x1024xf16, 1>,
                          %b: memref<1024x384xf16, 1>,
                          %y: memref<1x384xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<1x1024xf16, 1>, memref<1024x384xf16, 1>)
                   outs(%y : memref<1x384xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @dd_gemm_n192
  // CHECK: llvm.call @wrap_dd_matmul(
  func.func @dd_gemm_n192(%ctx: !hip.context,
                          %a: memref<1x1024xf16, 1>,
                          %b: memref<1024x192xf16, 1>,
                          %y: memref<1x192xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<1x1024xf16, 1>, memref<1024x192xf16, 1>)
                   outs(%y : memref<1x192xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @dd_matmul_n384
  // CHECK: llvm.call @wrap_dd_matmul(
  func.func @dd_matmul_n384(%ctx: !hip.context,
                            %a: memref<1x1024xf16, 1>,
                            %b: memref<1024x384xf16, 1>,
                            %y: memref<1x384xf16, 1>) {
    hip.matmul(%ctx) ins(%a, %b : memref<1x1024xf16, 1>, memref<1024x384xf16, 1>)
                     outs(%y : memref<1x384xf16, 1>)
    return
  }

  // ----- GPU fallbacks -------------------------------------------------------

  // CHECK-LABEL: llvm.func @gpu_gemm_unsupported_shape
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_gemm
  func.func @gpu_gemm_unsupported_shape(%ctx: !hip.context,
                                        %a: memref<128x256xf16, 1>,
                                        %b: memref<256x512xf16, 1>,
                                        %y: memref<128x512xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<128x256xf16, 1>, memref<256x512xf16, 1>)
                   outs(%y : memref<128x512xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @gpu_gemm_with_bias
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_gemm
  func.func @gpu_gemm_with_bias(%ctx: !hip.context,
                                %a: memref<1x1024xf16, 1>,
                                %b: memref<1024x384xf16, 1>,
                                %c: memref<384xf16, 1>,
                                %y: memref<1x384xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b, %c : memref<1x1024xf16, 1>, memref<1024x384xf16, 1>, memref<384xf16, 1>)
                   outs(%y : memref<1x384xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @gpu_gemm_transB
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_gemm
  func.func @gpu_gemm_transB(%ctx: !hip.context,
                             %a: memref<1x1024xf16, 1>,
                             %b: memref<384x1024xf16, 1>,
                             %y: memref<1x384xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<1x1024xf16, 1>, memref<384x1024xf16, 1>)
                   outs(%y : memref<1x384xf16, 1>)
                   {transB = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_gemm_alpha
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_gemm
  func.func @gpu_gemm_alpha(%ctx: !hip.context,
                            %a: memref<1x1024xf16, 1>,
                            %b: memref<1024x384xf16, 1>,
                            %y: memref<1x384xf16, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<1x1024xf16, 1>, memref<1024x384xf16, 1>)
                   outs(%y : memref<1x384xf16, 1>)
                   {alpha = 2.0 : f32}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_gemm_f32
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_gemm
  func.func @gpu_gemm_f32(%ctx: !hip.context,
                          %a: memref<1x1024xf32, 1>,
                          %b: memref<1024x384xf32, 1>,
                          %y: memref<1x384xf32, 1>) {
    hip.gemm(%ctx) ins(%a, %b : memref<1x1024xf32, 1>, memref<1024x384xf32, 1>)
                   outs(%y : memref<1x384xf32, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @gpu_matmul_unsupported_shape
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_hipblasLtMatmul
  func.func @gpu_matmul_unsupported_shape(%ctx: !hip.context,
                                          %a: memref<1x2048xf16, 1>,
                                          %b: memref<2048x384xf16, 1>,
                                          %y: memref<1x384xf16, 1>) {
    hip.matmul(%ctx) ins(%a, %b : memref<1x2048xf16, 1>, memref<2048x384xf16, 1>)
                     outs(%y : memref<1x384xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @gpu_matmul_rank3
  // CHECK-NOT: wrap_dd_matmul
  // CHECK: llvm.call @wrap_hipblasLtMatmul
  func.func @gpu_matmul_rank3(%ctx: !hip.context,
                              %a: memref<1x1x1024xf16, 1>,
                              %b: memref<1024x384xf16, 1>,
                              %y: memref<1x1x384xf16, 1>) {
    hip.matmul(%ctx) ins(%a, %b : memref<1x1x1024xf16, 1>, memref<1024x384xf16, 1>)
                     outs(%y : memref<1x1x384xf16, 1>)
    return
  }
}
