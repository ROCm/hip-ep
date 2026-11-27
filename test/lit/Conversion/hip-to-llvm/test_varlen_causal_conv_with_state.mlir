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
      %state: memref<2x4x9xf16, 1>,
      %output: memref<6x4xf16, 1>,
      %final: memref<2x4x9xf16, 1>) {
    hip.varlen_causal_conv_with_state(%ctx)
        ins(%input, %weight, %cumulative, %state :
            memref<6x4xf16, 1>, memref<4x1x4xf16, 1>, memref<3xi32, 1>,
            memref<2x4x9xf16, 1>)
        bias(%bias : memref<4xf16, 1>)
        outs(%output, %final :
             memref<6x4xf16, 1>, memref<2x4x9xf16, 1>)
        {activation = "silu", dilation = 3 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @varlen_conv_full
  // CHECK: %[[TOTAL:.*]] = llvm.mlir.constant(6 : i64) : i64
  // CHECK: %[[CH:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: %[[BATCH:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[K:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: %[[STLEN:.*]] = llvm.mlir.constant(9 : i64) : i64
  // CHECK: %[[DIL:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[ACT:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: llvm.call @wrap_varlen_causal_conv_with_state(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[TOTAL]], %[[CH]], %[[BATCH]], %[[K]], %[[DIL]], %[[STLEN]], %[[ACT]], %[[ELEM]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

  func.func @varlen_conv_no_bias_f32(
      %ctx: !hip.context,
      %input: memref<6x4xf32, 1>,
      %weight: memref<4x1x4xf32, 1>,
      %cumulative: memref<3xi32, 1>,
      %state: memref<2x4x6xf32, 1>,
      %output: memref<6x4xf32, 1>,
      %final: memref<2x4x6xf32, 1>) {
    hip.varlen_causal_conv_with_state(%ctx)
        ins(%input, %weight, %cumulative, %state :
            memref<6x4xf32, 1>, memref<4x1x4xf32, 1>, memref<3xi32, 1>,
            memref<2x4x6xf32, 1>)
        outs(%output, %final :
             memref<6x4xf32, 1>, memref<2x4x6xf32, 1>)
        {dilation = 2 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @varlen_conv_no_bias_f32
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[TOTAL2:.*]] = llvm.mlir.constant(6 : i64) : i64
  // CHECK: %[[CH2:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: %[[BATCH2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[K2:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: %[[STLEN2:.*]] = llvm.mlir.constant(6 : i64) : i64
  // CHECK: %[[DIL2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[ACT2:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[ELEM2:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: llvm.call @wrap_varlen_causal_conv_with_state(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NULL]], %{{.*}}, %{{.*}}, %{{.*}}, %[[TOTAL2]], %[[CH2]], %[[BATCH2]], %[[K2]], %[[DIL2]], %[[STLEN2]], %[[ACT2]], %[[ELEM2]])
}
