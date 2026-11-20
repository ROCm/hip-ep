// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.miopen.softmax lowers to a hip_miopen_softmax call whose last
// argument is the HIPDNN_EP_DATATYPE_* code of the element type, not its byte
// size: f16 and bf16 have the same size and need different kernels.
//
// Expected: hip_miopen_softmax(state, input_ptr, output_ptr,
//                              rows, cols, data_type)
// with rows = product of all but the last dim (1 for rank 1) and
// cols = the last dim. Codes: FLOAT = 0, HALF = 1, BFLOAT16 = 2.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK: llvm.func @hip_miopen_softmax(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)

module {
  // CHECK-LABEL: llvm.func @softmax_f16
  // CHECK: %[[COLS:.*]] = llvm.extractvalue %{{.*}}[3, 2]
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[D0:.*]] = llvm.extractvalue %{{.*}}[3, 0]
  // CHECK: %[[R0:.*]] = llvm.mul %[[ONE]], %[[D0]] : i64
  // CHECK: %[[D1:.*]] = llvm.extractvalue %{{.*}}[3, 1]
  // CHECK: %[[ROWS:.*]] = llvm.mul %[[R0]], %[[D1]] : i64
  // CHECK: %[[DT:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.call @hip_miopen_softmax(%{{.*}}, %{{.*}}, %{{.*}}, %[[ROWS]], %[[COLS]], %[[DT]])
  func.func @softmax_f16(%ctx: !hip.context,
                         %input: memref<2x8x64xf16, 1>,
                         %output: memref<2x8x64xf16, 1>) {
    hip.miopen.softmax(%ctx) ins(%input : memref<2x8x64xf16, 1>)
                             outs(%output : memref<2x8x64xf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @softmax_bf16
  // CHECK: llvm.mul
  // CHECK: %[[DT:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: llvm.call @hip_miopen_softmax(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[DT]])
  func.func @softmax_bf16(%ctx: !hip.context,
                          %input: memref<16x1500xbf16, 1>,
                          %output: memref<16x1500xbf16, 1>) {
    hip.miopen.softmax(%ctx) ins(%input : memref<16x1500xbf16, 1>)
                             outs(%output : memref<16x1500xbf16, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @softmax_f32
  // CHECK: llvm.mul
  // CHECK: %[[DT:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @hip_miopen_softmax(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[DT]])
  func.func @softmax_f32(%ctx: !hip.context,
                         %input: memref<4x4096xf32, 1>,
                         %output: memref<4x4096xf32, 1>) {
    hip.miopen.softmax(%ctx) ins(%input : memref<4x4096xf32, 1>)
                             outs(%output : memref<4x4096xf32, 1>)
    return
  }

  // A rank-1 input is a single row: rows must be the constant 1, not the
  // length of the only dimension.
  // CHECK-LABEL: llvm.func @softmax_rank1
  // CHECK: %[[COLS:.*]] = llvm.extractvalue %{{.*}}[3, 0]
  // CHECK: %[[ROWS:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NOT: llvm.mul
  // CHECK: llvm.call @hip_miopen_softmax(%{{.*}}, %{{.*}}, %{{.*}}, %[[ROWS]], %[[COLS]], %{{.*}})
  func.func @softmax_rank1(%ctx: !hip.context,
                           %input: memref<?xf32, 1>,
                           %output: memref<?xf32, 1>) {
    hip.miopen.softmax(%ctx) ins(%input : memref<?xf32, 1>)
                             outs(%output : memref<?xf32, 1>)
    return
  }
}
