// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// hip.geglu lowers to a single wrap_geglu call taking three buffers, a flat
// element count, the data-type code of the output element type (f32=0,
// f16=1, bf16=2, f64=6) and the gelu approximation (none=0, tanh=1).

module {
  func.func @geglu_tanh_f16(
      %ctx: !hip.context,
      %gate: memref<8x16xf16, 1>,
      %up: memref<8x16xf16, 1>,
      %output: memref<8x16xf16, 1>) {
    // CHECK-LABEL: llvm.func @geglu_tanh_f16
    // CHECK-DAG: llvm.mlir.constant(8 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(16 : i64) : i64
    // CHECK-DAG: %[[DTYPE:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: %[[TANH:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.call @wrap_geglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    hip.geglu(%ctx) ins(%gate, %up : memref<8x16xf16, 1>, memref<8x16xf16, 1>)
                    outs(%output : memref<8x16xf16, 1>) {approximate = "tanh"}
    return
  }

  func.func @geglu_erf_dynamic_f32(
      %ctx: !hip.context,
      %gate: memref<?x10240xf32, 1>,
      %up: memref<?x10240xf32, 1>,
      %output: memref<?x10240xf32, 1>) {
    // CHECK-LABEL: llvm.func @geglu_erf_dynamic_f32
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mlir.constant(10240 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(0 : i64) : i64
    // CHECK: llvm.call @wrap_geglu({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    hip.geglu(%ctx) ins(%gate, %up : memref<?x10240xf32, 1>, memref<?x10240xf32, 1>)
                    outs(%output : memref<?x10240xf32, 1>)
    return
  }
}
