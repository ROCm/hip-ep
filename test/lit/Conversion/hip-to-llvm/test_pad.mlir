// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.pad lowers to llvm.call @wrap_pad with the full 14-parameter
// signature:
//   (state, data, pads, cval_or_null, axes_or_null, out,
//    data_shape_ptr, data_rank,
//    out_shape_ptr,  out_rank,
//    pads_num_elements, axes_num_elements,
//    data_type, mode_id) -> i32.
// Mode is encoded as a small enum: 0=constant, 1=reflect, 2=edge, 3=wrap.
// Optional operands (cval, axes) are passed as null pointers when omitted.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Constant mode, no cval, no axes (both optional operands are null ptrs).
  func.func @pad_constant_default(
      %ctx: !hip.context,
      %data: memref<3x4xf32, 1>,
      %pads: memref<4xi64, 1>,
      %out: memref<5x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @pad_constant_default

    hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, 1>, memref<4xi64, 1>)
                  outs(%out : memref<5x6xf32, 1>)

    // Shape arrays for data + output (both rank 2).
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Constant mode with explicit constant_value scalar.
  func.func @pad_constant_with_cval(
      %ctx: !hip.context,
      %data: memref<3x4xf32, 1>,
      %pads: memref<4xi64, 1>,
      %cval: memref<f32, 1>,
      %out: memref<5x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @pad_constant_with_cval

    hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, 1>, memref<4xi64, 1>)
                  cval(%cval : memref<f32, 1>)
                  outs(%out : memref<5x6xf32, 1>)

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Reflect mode (mode_id = 1).
  func.func @pad_reflect(
      %ctx: !hip.context,
      %data: memref<3x4xf32, 1>,
      %pads: memref<4xi64, 1>,
      %out: memref<5x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @pad_reflect

    hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, 1>, memref<4xi64, 1>)
                  outs(%out : memref<5x6xf32, 1>) {mode = "reflect"}

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }

  // With axes operand present.
  func.func @pad_axes(
      %ctx: !hip.context,
      %data: memref<3x4xf32, 1>,
      %pads: memref<2xi64, 1>,
      %axes: memref<1xi64, 1>,
      %out: memref<3x6xf32, 1>) {
    // CHECK-LABEL: llvm.func @pad_axes

    hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, 1>, memref<2xi64, 1>)
                  axes(%axes : memref<1xi64, 1>)
                  outs(%out : memref<3x6xf32, 1>)

    // CHECK: llvm.call @wrap_pad({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }
}
