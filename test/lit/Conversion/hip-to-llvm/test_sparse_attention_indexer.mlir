// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @sai(
      %ctx: !hip.context,
      %qk: memref<?x?x640xf16, 1>,
      %qw: memref<128xf16, 1>,
      %kw: memref<128xf16, 1>,
      %cos: memref<?x?x64xf16, 1>,
      %sin: memref<?x?x64xf16, 1>,
      %past: memref<?x?x128xf16, 1>,
      %mask: memref<?x?xi64, 1>,
      %sel: memref<?x?x2051xi32, 1>,
      %present: memref<?x?x128xf16, 1>) {
    hip.sparse_attention_indexer(%ctx)
        ins(%qk, %qw, %kw, %cos, %sin, %past :
            memref<?x?x640xf16, 1>, memref<128xf16, 1>, memref<128xf16, 1>,
            memref<?x?x64xf16, 1>, memref<?x?x64xf16, 1>,
            memref<?x?x128xf16, 1>)
        mask(%mask : memref<?x?xi64, 1>)
        outs(%sel, %present : memref<?x?x2051xi32, 1>, memref<?x?x128xf16, 1>)
        {compress_ratio = 4 : i64, token_budget = 2048 : i64,
         epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
    // CHECK-LABEL: llvm.func @sai(
    // CHECK: %[[B:.*]] = llvm.extractvalue %[[Q:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[S:.*]] = llvm.extractvalue %[[Q]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[P:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[N:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK: %[[D:.*]] = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %[[R:.*]] = llvm.mlir.constant(64 : i64) : i64
    // CHECK: %[[X:.*]] = llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[BATCHED:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[RATIO:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK: %[[BUDGET:.*]] = llvm.mlir.constant(2048 : i64) : i64
    // CHECK: %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(0.0883883461 : f32) : f32
    // CHECK: llvm.call @wrap_sparse_attention_indexer(%{{.*}}, %[[B]], %[[S]], %[[P]], %[[N]], %[[D]], %[[R]], %[[X]], %[[BATCHED]], %[[RATIO]], %[[BUDGET]], %[[ELEM]], %[[EPS]], %[[SCALE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32) -> i32
    return
  }

  func.func @sai_no_mask(
      %ctx: !hip.context,
      %qk: memref<2x3x384xf32, 1>,
      %qw: memref<128xf32, 1>,
      %kw: memref<128xf32, 1>,
      %cos: memref<64x32xf32, 1>,
      %sin: memref<64x32xf32, 1>,
      %past: memref<2x5x128xf32, 1>,
      %sel: memref<2x3x67xi32, 1>,
      %present: memref<2x8x128xf32, 1>) {
    hip.sparse_attention_indexer(%ctx)
        ins(%qk, %qw, %kw, %cos, %sin, %past :
            memref<2x3x384xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>,
            memref<64x32xf32, 1>, memref<64x32xf32, 1>,
            memref<2x5x128xf32, 1>)
        outs(%sel, %present : memref<2x3x67xi32, 1>, memref<2x8x128xf32, 1>)
        {compress_ratio = 4 : i64, token_budget = 64 : i64,
         epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
    // CHECK-LABEL: llvm.func @sai_no_mask(
    // CHECK: %[[NOMASK:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[X:.*]] = llvm.mlir.constant(64 : i64) : i64
    // CHECK-NEXT: %[[BATCHED:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK-NEXT: %[[RATIO:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK-NEXT: %[[BUDGET:.*]] = llvm.mlir.constant(64 : i64) : i64
    // CHECK-NEXT: %[[ELEM:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK: llvm.call @wrap_sparse_attention_indexer(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NOMASK]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[X]], %[[BATCHED]], %[[RATIO]], %[[BUDGET]], %[[ELEM]], %{{.*}}, %{{.*}})
    return
  }
}
