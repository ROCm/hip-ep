// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // hip.add with quantized `quant_types` (from hip-legalize-quant-types)
  // lowers to wrap_qelementwise; scales and zero points come from the types.
  func.func @add_quant_types_i8(
      %ctx: !hip.context,
      %lhs: memref<1x128x32xi8, 1>,
      %rhs: memref<1x128x32xi8, 1>,
      %out: memref<1x128x32xi8, 1>) {
    // CHECK-LABEL: llvm.func @add_quant_types_i8
    // CHECK-NOT: @wrap_elementwise
    // CHECK-DAG: %[[MA:.*]] = llvm.mlir.constant(5.000000e-01 : f32)
    // CHECK-DAG: %[[ZA:.*]] = llvm.mlir.constant(-5 : i64)
    // CHECK-DAG: %[[MB:.*]] = llvm.mlir.constant(2.500000e-01 : f32)
    // CHECK-DAG: %[[ZO:.*]] = llvm.mlir.constant(7 : i64)
    // CHECK: llvm.call @wrap_qelementwise({{.*}}, %[[MA]], %[[ZA]], %[[MB]], %{{.*}}, %[[ZO]])
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xi8, 1>, memref<1x128x32xi8, 1>)
                  outs(%out : memref<1x128x32xi8, 1>)
                  {quant_types = [!quant.uniform<i8:f32, 0.1:-5>,
                                  !quant.uniform<i8:f32, 0.05:3>,
                                  !quant.uniform<i8:f32, 0.2:7>]}
    return
  }

  // The same op with non-quantized `quant_types` (as stamped by
  // hip-legalize-quant-types) or none at all is the float add.
  func.func @add_f32(
      %ctx: !hip.context,
      %lhs: memref<1x128x32xf32, 1>,
      %rhs: memref<1x128x32xf32, 1>,
      %out: memref<1x128x32xf32, 1>) {
    // CHECK-LABEL: llvm.func @add_f32
    // CHECK-NOT: @wrap_qelementwise
    // CHECK: llvm.call @wrap_elementwise
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xf32, 1>, memref<1x128x32xf32, 1>)
                  outs(%out : memref<1x128x32xf32, 1>)
    return
  }

  func.func @add_f32_quant_types(
      %ctx: !hip.context,
      %lhs: memref<1x128x32xf32, 1>,
      %rhs: memref<1x128x32xf32, 1>,
      %out: memref<1x128x32xf32, 1>) {
    // CHECK-LABEL: llvm.func @add_f32_quant_types
    // CHECK-NOT: @wrap_qelementwise
    // CHECK: llvm.call @wrap_elementwise
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xf32, 1>, memref<1x128x32xf32, 1>)
                  outs(%out : memref<1x128x32xf32, 1>)
                  {quant_types = [f32, f32, f32]}
    return
  }

  // An integer add is not quantized: the ui16 data alone must not select the
  // quantized kernel.
  func.func @add_i32_quant_types(
      %ctx: !hip.context,
      %lhs: memref<1x128x32xi32, 1>,
      %rhs: memref<1x128x32xi32, 1>,
      %out: memref<1x128x32xi32, 1>) {
    // CHECK-LABEL: llvm.func @add_i32_quant_types
    // CHECK-NOT: @wrap_qelementwise
    // CHECK: llvm.call @wrap_elementwise
    hip.add(%ctx) ins(%lhs, %rhs : memref<1x128x32xi32, 1>, memref<1x128x32xi32, 1>)
                  outs(%out : memref<1x128x32xi32, 1>)
                  {quant_types = [i32, i32, i32]}
    return
  }
}
