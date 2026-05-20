// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.and lowers to llvm.call @wrap_and with signature
//   (state, lhs, rhs, output, num_elements, data_type) -> i32.
// Both inputs are i1 (boolean). Since i1 is not in the HIPDNN dtype enum,
// the lowering passes 0 as a sentinel.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @and_static_1d(
      %ctx: !hip.context,
      %a: memref<64xi1, 1>,
      %b: memref<64xi1, 1>,
      %c: memref<64xi1, 1>) {
    // CHECK-LABEL: llvm.func @and_static_1d

    hip.and(%ctx) ins(%a, %b : memref<64xi1, 1>, memref<64xi1, 1>)
                  outs(%c : memref<64xi1, 1>)

    // CHECK: llvm.call @wrap_and({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @and_dynamic_2d(
      %ctx: !hip.context,
      %a: memref<?x?xi1, 1>,
      %b: memref<?x?xi1, 1>,
      %c: memref<?x?xi1, 1>) {
    // CHECK-LABEL: llvm.func @and_dynamic_2d

    hip.and(%ctx) ins(%a, %b : memref<?x?xi1, 1>, memref<?x?xi1, 1>)
                  outs(%c : memref<?x?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_and({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
