// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @dsa(
      %ctx: !hip.context,
      %q: memref<?x?x6144xf16, 1>,
      %k: memref<?x?x512xf16, 1>,
      %v: memref<?x?x512xf16, 1>,
      %past_k: memref<?x2x?x?xf16, 1>,
      %past_v: memref<?x2x?x?xf16, 1>,
      %idx: memref<?x2051xi32, 1>,
      %cnt: memref<?xi32, 1>,
      %seqlens: memref<?xi32, 1>,
      %out: memref<?x?x6144xf16, 1>,
      %present_k: memref<?x2x?x?xf16, 1>,
      %present_v: memref<?x2x?x?xf16, 1>) {
    hip.dynamic_sparse_attention(%ctx)
        ins(%q, %k, %v, %past_k, %past_v, %idx, %cnt, %seqlens :
            memref<?x?x6144xf16, 1>, memref<?x?x512xf16, 1>,
            memref<?x?x512xf16, 1>, memref<?x2x?x?xf16, 1>,
            memref<?x2x?x?xf16, 1>, memref<?x2051xi32, 1>,
            memref<?xi32, 1>, memref<?xi32, 1>)
        outs(%out, %present_k, %present_v : memref<?x?x6144xf16, 1>,
             memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 2 : i64, scale = 6.250000e-02 : f32}
    // CHECK-LABEL: llvm.func @dsa
    // CHECK: %[[B:.*]] = llvm.extractvalue %[[Q:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[S:.*]] = llvm.extractvalue %[[Q]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[N:.*]] = llvm.mlir.constant(24 : i64) : i64
    // CHECK: %[[KVN:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[D:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK: %[[CAP:.*]] = llvm.extractvalue %[[PK:.*]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<4 x i64>, array<4 x i64>)>
    // CHECK: %[[K:.*]] = llvm.mlir.constant(2051 : i64) : i64
    // CHECK: %[[PKB:.*]] = llvm.extractvalue %[[PK]][3, 0]
    // CHECK: %[[PKH:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[PKD:.*]] = llvm.extractvalue %[[PK]][3, 3]
    // CHECK: %[[PVB:.*]] = llvm.extractvalue %[[PV:.*]][3, 0]
    // CHECK: %[[PVH:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[PVC:.*]] = llvm.extractvalue %[[PV]][3, 2]
    // CHECK: %[[PVD:.*]] = llvm.extractvalue %[[PV]][3, 3]
    // CHECK: %[[OKC:.*]] = llvm.extractvalue %[[OK:.*]][3, 2]
    // CHECK: %[[OKD:.*]] = llvm.extractvalue %[[OK]][3, 3]
    // CHECK: %[[OVC:.*]] = llvm.extractvalue %[[OV:.*]][3, 2]
    // CHECK: %[[OVD:.*]] = llvm.extractvalue %[[OV]][3, 3]
    // CHECK: %[[IDXR:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK: %[[CNTR:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[SEQR:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
    // CHECK: llvm.call @wrap_dynamic_sparse_attention(%{{.*}}, %[[B]], %[[S]], %[[N]], %[[KVN]], %[[D]], %[[CAP]], %[[K]], %[[PKB]], %[[PKH]], %[[PKD]], %[[PVB]], %[[PVH]], %[[PVC]], %[[PVD]], %[[OKC]], %[[OKD]], %[[OVC]], %[[OVD]], %[[IDXR]], %[[CNTR]], %[[SEQR]], %[[SCALE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32) -> i32
    return
  }
}
