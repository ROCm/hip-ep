// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @quant_per_tensor_u8(
      %ctx: !hip.context,
      %x: memref<4x8xf32, 1>,
      %scale: memref<f32, 1>,
      %zp: memref<ui8, 1>,
      %output: memref<4x8xui8, 1>) {
    // CHECK-LABEL: llvm.func @quant_per_tensor_u8
    hip.quantize_linear(%ctx)
        ins(%x, %scale : memref<4x8xf32, 1>, memref<f32, 1>)
        zero_points(%zp : memref<ui8, 1>)
        outs(%output : memref<4x8xui8, 1>)
        {axis = 1 : i64, block_size = 0 : i64, output_dtype = 0 : i64,
         precision = 0 : i64, saturate = 1 : i64}
    // CHECK: llvm.call @wrap_quantize_linear
    return
  }

  func.func @quant_dynamic_2d_u8(
      %ctx: !hip.context,
      %x: memref<?x?xf32, 1>,
      %scale: memref<?xf32, 1>,
      %output: memref<?x?xui8, 1>) {
    // CHECK-LABEL: llvm.func @quant_dynamic_2d_u8
    hip.quantize_linear(%ctx)
        ins(%x, %scale : memref<?x?xf32, 1>, memref<?xf32, 1>)
        outs(%output : memref<?x?xui8, 1>)
        {axis = 1 : i64, block_size = 0 : i64, output_dtype = 0 : i64,
         precision = 0 : i64, saturate = 1 : i64}
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.call @wrap_quantize_linear
    return
  }
}
