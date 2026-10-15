// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.isnan lowers to llvm.call @wrap_isnan with signature
//   (state, input_ptr, output_ptr, num_elements, data_type) -> i32.
// data_type is the floating-point input type. The output is a 1-byte boolean
// (ui8 from the ORT frontend, or i1 in hand-written IR).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @isnan_static_f16_ui8(
      %ctx: !hip.context,
      %x: memref<24x20x1x1xf16, 1>,
      %y: memref<24x20x1x1xui8, 1>) {
    // CHECK-LABEL: llvm.func @isnan_static_f16_ui8

    hip.isnan(%ctx) ins(%x : memref<24x20x1x1xf16, 1>)
                    outs(%y : memref<24x20x1x1xui8, 1>)

    // CHECK: llvm.call @wrap_isnan({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }

  func.func @isnan_dynamic_f32_i1(
      %ctx: !hip.context,
      %x: memref<?x?xf32, 1>,
      %y: memref<?x?xi1, 1>) {
    // CHECK-LABEL: llvm.func @isnan_dynamic_f32_i1

    hip.isnan(%ctx) ins(%x : memref<?x?xf32, 1>)
                    outs(%y : memref<?x?xi1, 1>)

    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_isnan({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    return
  }
}
