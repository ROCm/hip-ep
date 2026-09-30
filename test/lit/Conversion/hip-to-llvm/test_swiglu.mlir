// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// hip.swiglu lowers to a single wrap_swiglu call taking three buffers plus a
// flat element count; the data-type code comes from the output element type
// (f32=0, f16=1, bf16=2, f64=6).

module {
  func.func @swiglu_static_f16(
      %ctx: !hip.context,
      %gate: memref<8x16xf16, 1>,
      %up: memref<8x16xf16, 1>,
      %output: memref<8x16xf16, 1>) {
    // CHECK-LABEL: llvm.func @swiglu_static_f16
    // CHECK-DAG: llvm.mlir.constant(8 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(16 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.call @wrap_swiglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    hip.swiglu(%ctx) ins(%gate, %up : memref<8x16xf16, 1>, memref<8x16xf16, 1>)
                     outs(%output : memref<8x16xf16, 1>)
    return
  }

  func.func @swiglu_dynamic_f32(
      %ctx: !hip.context,
      %gate: memref<?x14336xf32, 1>,
      %up: memref<?x14336xf32, 1>,
      %output: memref<?x14336xf32, 1>) {
    // CHECK-LABEL: llvm.func @swiglu_dynamic_f32
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mlir.constant(14336 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(0 : i64) : i64
    // CHECK: llvm.call @wrap_swiglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    hip.swiglu(%ctx) ins(%gate, %up : memref<?x14336xf32, 1>, memref<?x14336xf32, 1>)
                     outs(%output : memref<?x14336xf32, 1>)
    return
  }

  func.func @swiglu_bf16(
      %ctx: !hip.context,
      %gate: memref<32xbf16, 1>,
      %up: memref<32xbf16, 1>,
      %output: memref<32xbf16, 1>) {
    // CHECK-LABEL: llvm.func @swiglu_bf16
    // CHECK-DAG: llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_swiglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    hip.swiglu(%ctx) ins(%gate, %up : memref<32xbf16, 1>, memref<32xbf16, 1>)
                     outs(%output : memref<32xbf16, 1>)
    return
  }

  func.func @swiglu_f64(
      %ctx: !hip.context,
      %gate: memref<8xf64, 1>,
      %up: memref<8xf64, 1>,
      %output: memref<8xf64, 1>) {
    // CHECK-LABEL: llvm.func @swiglu_f64
    // CHECK-DAG: llvm.mlir.constant(6 : i64) : i64
    // CHECK: llvm.call @wrap_swiglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    hip.swiglu(%ctx) ins(%gate, %up : memref<8xf64, 1>, memref<8xf64, 1>)
                     outs(%output : memref<8xf64, 1>)
    return
  }
}
