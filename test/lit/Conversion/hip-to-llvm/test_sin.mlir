// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.sin lowers to llvm.call @wrap_sin with signature
//   (state, input_ptr, output_ptr, num_elements, data_type) -> i32.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @sin_static_1d_f32(
      %ctx: !hip.context,
      %x: memref<256xf32, 1>,
      %y: memref<256xf32, 1>) {
    // CHECK-LABEL: llvm.func @sin_static_1d_f32

    hip.sin(%ctx) ins(%x : memref<256xf32, 1>)
                  outs(%y : memref<256xf32, 1>)

    // CHECK: llvm.call @wrap_sin({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @sin_dynamic_2d_f16(
      %ctx: !hip.context,
      %x: memref<?x?xf16, 1>,
      %y: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @sin_dynamic_2d_f16

    hip.sin(%ctx) ins(%x : memref<?x?xf16, 1>)
                  outs(%y : memref<?x?xf16, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_sin({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
