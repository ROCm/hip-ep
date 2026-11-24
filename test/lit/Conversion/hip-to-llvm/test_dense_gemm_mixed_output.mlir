// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @half_score(%ctx: !hip.context, %a: memref<2x4xf16, 1>, %b: memref<4x8xf16, 1>, %y: memref<2x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @half_score
    // CHECK: llvm.call @wrap_matmul_typed
    hip.matmul(%ctx) ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>) outs(%y : memref<2x8xf32, 1>)
    return
  }
  func.func @bf16_bias(%ctx: !hip.context, %a: memref<2x4xbf16, 1>, %b: memref<4x8xbf16, 1>, %c: memref<8xf32, 1>, %y: memref<2x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @bf16_bias
    // CHECK: llvm.call @wrap_gemm_typed
    hip.gemm(%ctx) ins(%a, %b, %c : memref<2x4xbf16, 1>, memref<4x8xbf16, 1>, memref<8xf32, 1>) outs(%y : memref<2x8xf32, 1>)
    return
  }
  func.func @broadcast_score(%ctx: !hip.context, %a: memref<2x1x2x4xf16, 1>, %b: memref<1x3x4x8xf16, 1>, %y: memref<2x3x2x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @broadcast_score
    // CHECK: llvm.call @wrap_matmul_broadcast_typed
    hip.matmul(%ctx) ins(%a, %b : memref<2x1x2x4xf16, 1>, memref<1x3x4x8xf16, 1>) outs(%y : memref<2x3x2x8xf32, 1>)
    return
  }
}
