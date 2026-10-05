// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.qlinear_conv -> wrap_qlinear_conv
//
// UINT8 activations are HIPDNN_EP_DATATYPE_UINT8 = 7, INT8 weights are
// HIPDNN_EP_DATATYPE_INT8 = 5, and an i32 bias is HIPDNN_EP_DATATYPE_INT32 = 3.
// A missing bias passes a null pointer and the unsupported sentinel -1.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @qlinear_conv_stride2
  // CHECK-DAG: llvm.mlir.constant(4 : i64)
  // CHECK-DAG: llvm.mlir.constant(8 : i64)
  // CHECK-DAG: llvm.mlir.constant(7 : i64)
  // CHECK-DAG: llvm.mlir.constant(5 : i64)
  // CHECK: llvm.call @wrap_qlinear_conv
  func.func @qlinear_conv_stride2(
      %ctx: !hip.context,
      %x: memref<1x3x8x8xui8, 1>, %xs: memref<f32, 1>, %xz: memref<ui8, 1>,
      %w: memref<4x3x3x3xi8, 1>, %ws: memref<f32, 1>, %wz: memref<i8, 1>,
      %ys: memref<f32, 1>, %yz: memref<ui8, 1>, %b: memref<4xi32, 1>,
      %y: memref<1x4x4x4xui8, 1>) {
    hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
        memref<1x3x8x8xui8, 1>, memref<f32, 1>, memref<ui8, 1>,
        memref<4x3x3x3xi8, 1>, memref<f32, 1>, memref<i8, 1>,
        memref<f32, 1>, memref<ui8, 1>, memref<4xi32, 1>)
      outs(%y : memref<1x4x4x4xui8, 1>) {
        dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
        pads = [1, 1, 1, 1], strides = [2, 2]
      }
    return
  }

  // CHECK-LABEL: llvm.func @qlinear_conv_no_bias
  // CHECK-DAG: llvm.mlir.constant(-1 : i64)
  // CHECK-DAG: llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.call @wrap_qlinear_conv
  func.func @qlinear_conv_no_bias(
      %ctx: !hip.context,
      %x: memref<1x4x5x5xui8, 1>, %xs: memref<f32, 1>, %xz: memref<ui8, 1>,
      %w: memref<6x4x1x1xi8, 1>, %ws: memref<f32, 1>, %wz: memref<i8, 1>,
      %ys: memref<f32, 1>, %yz: memref<ui8, 1>,
      %y: memref<1x6x5x5xui8, 1>) {
    hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
        memref<1x4x5x5xui8, 1>, memref<f32, 1>, memref<ui8, 1>,
        memref<6x4x1x1xi8, 1>, memref<f32, 1>, memref<i8, 1>,
        memref<f32, 1>, memref<ui8, 1>)
      outs(%y : memref<1x6x5x5xui8, 1>) {
        dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
        pads = [0, 0, 0, 0], strides = [1, 1]
      }
    return
  }
}
