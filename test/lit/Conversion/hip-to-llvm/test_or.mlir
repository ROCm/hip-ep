// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @or_static_1d(
      %ctx: !hip.context,
      %a: memref<64xi1, 1>,
      %b: memref<64xi1, 1>,
      %c: memref<64xi1, 1>) {
    // CHECK-LABEL: llvm.func @or_static_1d

    hip.or(%ctx) ins(%a, %b : memref<64xi1, 1>, memref<64xi1, 1>)
                 outs(%c : memref<64xi1, 1>)

    // CHECK: llvm.call @wrap_or({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @or_dynamic_2d(
      %ctx: !hip.context,
      %a: memref<?x?xi1, 1>,
      %b: memref<?x?xi1, 1>,
      %c: memref<?x?xi1, 1>) {
    // CHECK-LABEL: llvm.func @or_dynamic_2d

    hip.or(%ctx) ins(%a, %b : memref<?x?xi1, 1>, memref<?x?xi1, 1>)
                 outs(%c : memref<?x?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_or({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
