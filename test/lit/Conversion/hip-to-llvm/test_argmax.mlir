// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // Text-encoder ArgMax: dynamic rank-2 i32, last axis dropped.
  func.func @argmax_dynamic_2d_i32(
      %ctx: !hip.context,
      %data: memref<?x?xi32, 1>,
      %indices: memref<?xi64, 1>) {
    // CHECK-LABEL: llvm.func @argmax_dynamic_2d_i32
    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_arg_max({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64) -> i32

    hip.arg_max(%ctx)
        ins(%data : memref<?x?xi32, 1>)
        outs(%indices : memref<?xi64, 1>)
        {axis = 1 : i64, keepdims = 0 : i64, select_last_index = 0 : i64}
    return
  }

  func.func @argmax_static_keepdims_i32(
      %ctx: !hip.context,
      %data: memref<4x8xi32, 1>,
      %indices: memref<4x1xi64, 1>) {
    // CHECK-LABEL: llvm.func @argmax_static_keepdims_i32
    // CHECK: llvm.call @wrap_arg_max({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64) -> i32

    hip.arg_max(%ctx)
        ins(%data : memref<4x8xi32, 1>)
        outs(%indices : memref<4x1xi64, 1>)
        {axis = 1 : i64, keepdims = 1 : i64, select_last_index = 1 : i64}
    return
  }
}
