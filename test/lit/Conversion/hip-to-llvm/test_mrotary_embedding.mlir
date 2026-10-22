// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @mrope(
      %ctx: !hip.context,
      %x: memref<?x?x6144xf16, 1>,
      %pos: memref<3x?x?xi64, 1>,
      %cos: memref<262144x32xf16, 1>,
      %sin: memref<262144x32xf16, 1>,
      %out: memref<?x?x6144xf16, 1>) {
    hip.mrotary_embedding(%ctx)
        ins(%x, %pos, %cos, %sin :
            memref<?x?x6144xf16, 1>, memref<3x?x?xi64, 1>,
            memref<262144x32xf16, 1>, memref<262144x32xf16, 1>)
        outs(%out : memref<?x?x6144xf16, 1>)
        {mrope_layout = 1 : i64, mrope_section = [11, 11, 10],
         num_heads = 24 : i64, rotary_embedding_dim = 64 : i64,
         scale = 1.0 : f32}
    // CHECK-LABEL: llvm.func @mrope
    // CHECK: %[[B:.*]] = llvm.extractvalue %[[X:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[S:.*]] = llvm.extractvalue %[[X]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[NH:.*]] = llvm.mlir.constant(24 : i64) : i64
    // CHECK: %[[HD:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK: %[[RD:.*]] = llvm.mlir.constant(64 : i64) : i64
    // CHECK: %[[MAXP:.*]] = llvm.mlir.constant(262144 : i64) : i64
    // CHECK: %[[ST:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK: %[[SH:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK: %[[SW:.*]] = llvm.mlir.constant(10 : i64) : i64
    // CHECK: %[[LAYOUT:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    // CHECK: %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_mrotary_embedding(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[B]], %[[S]], %[[NH]], %[[HD]], %[[RD]], %[[MAXP]], %[[ST]], %[[SH]], %[[SW]], %[[LAYOUT]], %[[SCALE]], %[[ELEM]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
    return
  }
}
