// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.mod lowers to llvm.call @wrap_mod with signature
//   (state, lhs, rhs, output, num_elements, data_type, fmod) -> i32.
// `fmod` attribute (default 0 = integer remainder, 1 = fmod for floats) is
// forwarded as the trailing i64 argument.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Integer remainder (fmod = 0).
  func.func @mod_int_static(
      %ctx: !hip.context,
      %a: memref<3x4xi32, 1>,
      %b: memref<3x4xi32, 1>,
      %c: memref<3x4xi32, 1>) {
    // CHECK-LABEL: llvm.func @mod_int_static

    hip.mod(%ctx) ins(%a, %b : memref<3x4xi32, 1>, memref<3x4xi32, 1>)
                  outs(%c : memref<3x4xi32, 1>)

    // CHECK: llvm.call @wrap_mod({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }

  // Float fmod (fmod = 1).
  func.func @mod_float_fmod(
      %ctx: !hip.context,
      %a: memref<3x4xf32, 1>,
      %b: memref<3x4xf32, 1>,
      %c: memref<3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @mod_float_fmod

    hip.mod(%ctx) ins(%a, %b : memref<3x4xf32, 1>, memref<3x4xf32, 1>)
                  outs(%c : memref<3x4xf32, 1>) {fmod = 1 : i64}

    // CHECK: llvm.call @wrap_mod({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }

  // Dynamic shapes.
  func.func @mod_dynamic(
      %ctx: !hip.context,
      %a: memref<?x?xi32, 1>,
      %b: memref<?x?xi32, 1>,
      %c: memref<?x?xi32, 1>) {
    // CHECK-LABEL: llvm.func @mod_dynamic

    hip.mod(%ctx) ins(%a, %b : memref<?x?xi32, 1>, memref<?x?xi32, 1>)
                  outs(%c : memref<?x?xi32, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_mod({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }
}
