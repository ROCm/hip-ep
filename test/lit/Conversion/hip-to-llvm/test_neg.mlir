// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.neg lowers to llvm.call @wrap_neg with signature
//   (state, input_ptr, output_ptr, num_elements, data_type) -> i32.
// Covers static and dynamic shapes; dtype enum 0 = f32, 1 = f16.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Static 2D f32: num_elements folds to 128*64 = 8192 at compile time.
  func.func @neg_static_2d_f32(
      %ctx: !hip.context,
      %x: memref<128x64xf32, 1>,
      %y: memref<128x64xf32, 1>) {
    // CHECK-LABEL: llvm.func @neg_static_2d_f32

    hip.neg(%ctx) ins(%x : memref<128x64xf32, 1>)
                  outs(%y : memref<128x64xf32, 1>)

    // CHECK: llvm.call @wrap_neg({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  // Dynamic 2D f16: num_elements computed via llvm.extractvalue * llvm.mul.
  func.func @neg_dynamic_2d_f16(
      %ctx: !hip.context,
      %x: memref<?x?xf16, 1>,
      %y: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @neg_dynamic_2d_f16

    hip.neg(%ctx) ins(%x : memref<?x?xf16, 1>)
                  outs(%y : memref<?x?xf16, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: llvm.call @wrap_neg({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
