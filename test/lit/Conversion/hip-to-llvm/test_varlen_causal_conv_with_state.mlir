// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @varlen_conv_full(
      %ctx: !hip.context,
      %input: memref<6x4xf16, 1>,
      %weight: memref<4x1x4xf16, 1>,
      %cumulative: memref<3xi32, 1>,
      %bias: memref<4xf16, 1>,
      %state: memref<2x4x3xf16, 1>,
      %output: memref<6x4xf16, 1>,
      %present: memref<2x4x3xf16, 1>) {
    hip.varlen_causal_conv_with_state(%ctx)
        ins(%input, %weight, %cumulative, %bias, %state :
            memref<6x4xf16, 1>, memref<4x1x4xf16, 1>, memref<3xi32, 1>,
            memref<4xf16, 1>, memref<2x4x3xf16, 1>)
        outs(%output, %present :
             memref<6x4xf16, 1>, memref<2x4x3xf16, 1>)
        {activation = "silu", dilation = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @varlen_conv_full
  // CHECK: llvm.call @wrap_varlen_causal_conv_with_state({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
}
