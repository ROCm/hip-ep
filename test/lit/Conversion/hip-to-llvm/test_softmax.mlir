// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// The standalone Softmax runtime receives semantic dtype plus both descriptors.
// Every returned status must feed the shared ORT-visible recorder.
// CHECK-LABEL: llvm.func @softmax_f16
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: %[[F16_STATUS:.*]] = llvm.call @hip_miopen_softmax({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[F16_STATUS]])
func.func @softmax_f16(%ctx: !hip.context,
                       %input: memref<2x4xf16, 1>,
                       %output: memref<2x4xf16, 1>) {
  hip.miopen.softmax(%ctx)
      ins(%input : memref<2x4xf16, 1>)
      outs(%output : memref<2x4xf16, 1>)
  return
}

// CHECK-LABEL: llvm.func @softmax_f32
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: %[[F32_STATUS:.*]] = llvm.call @hip_miopen_softmax({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[F32_STATUS]])
func.func @softmax_f32(%ctx: !hip.context,
                       %input: memref<2x4xf32, 1>,
                       %output: memref<2x4xf32, 1>) {
  hip.miopen.softmax(%ctx)
      ins(%input : memref<2x4xf32, 1>)
      outs(%output : memref<2x4xf32, 1>)
  return
}

// CHECK-LABEL: llvm.func @softmax_bf16
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: %[[BF16_STATUS:.*]] = llvm.call @hip_miopen_softmax({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT: llvm.call @hipdnn_ep_state_record_status({{.*}}, %[[BF16_STATUS]])
func.func @softmax_bf16(%ctx: !hip.context,
                        %input: memref<2x4xbf16, 1>,
                        %output: memref<2x4xbf16, 1>) {
  hip.miopen.softmax(%ctx)
      ins(%input : memref<2x4xbf16, 1>)
      outs(%output : memref<2x4xbf16, 1>)
  return
}
