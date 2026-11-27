// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.skip_layer_norm lowers to one llvm.call @wrap_skip_layer_norm; absent
// beta / bias / sum operands become null pointers. The skip's element count
// is passed next to the input's, so the runtime can repeat a skip that holds
// only the trailing rows. round_sum is passed as a trailing i32.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @skip_layer_norm_full
  func.func @skip_layer_norm_full(
      %ctx: !hip.context,
      %input: memref<1x128x768xf16, 1>,
      %skip: memref<1x128x768xf16, 1>,
      %gamma: memref<768xf16, 1>,
      %beta: memref<768xf16, 1>,
      %bias: memref<768xf16, 1>,
      %output: memref<1x128x768xf16, 1>,
      %sum: memref<1x128x768xf16, 1>) {
    hip.skip_layer_norm(%ctx)
        ins(%input, %skip, %gamma : memref<1x128x768xf16, 1>,
            memref<1x128x768xf16, 1>, memref<768xf16, 1>,
            beta %beta : memref<768xf16, 1>, bias %bias : memref<768xf16, 1>)
        outs(%output, %sum : memref<1x128x768xf16, 1>, memref<1x128x768xf16, 1>)
        {epsilon = 9.99999974E-6 : f32}
    // CHECK: llvm.mlir.constant(9.99999974E-6 : f32) : f32
    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.call @wrap_skip_layer_norm({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i32) -> i32
    return
  }

  // CHECK-LABEL: llvm.func @skip_layer_norm_minimal
  func.func @skip_layer_norm_minimal(
      %ctx: !hip.context,
      %input: memref<?x768xf32, 1>,
      %skip: memref<?x768xf32, 1>,
      %gamma: memref<768xf32, 1>,
      %output: memref<?x768xf32, 1>) {
    hip.skip_layer_norm(%ctx)
        ins(%input, %skip, %gamma : memref<?x768xf32, 1>,
            memref<?x768xf32, 1>, memref<768xf32, 1>)
        outs(%output : memref<?x768xf32, 1>)
        {epsilon = 9.99999974E-6 : f32, round_sum}
    // CHECK: llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.mlir.constant(4 : i64) : i64
    // CHECK: llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.call @wrap_skip_layer_norm
    return
  }
}
