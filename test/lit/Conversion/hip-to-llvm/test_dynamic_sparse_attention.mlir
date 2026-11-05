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
    hip.dynamic_sparse_attention(%ctx) ins(%q : memref<?x?x6144xf16, 1>)
        kv(%k, %v : memref<?x?x512xf16, 1>, memref<?x?x512xf16, 1>)
        cache(%past_k, %past_v : memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        select(%idx, %cnt, %seqlens : memref<?x2051xi32, 1>, memref<?xi32, 1>,
               memref<?xi32, 1>)
        outs(%out, %present_k, %present_v : memref<?x?x6144xf16, 1>,
             memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 2 : i64, scale = 6.250000e-02 : f32}
    // CHECK-LABEL: llvm.func @dsa
    // CHECK: %[[Q:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
    // CHECK: %[[KP:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
    // CHECK: %[[VP:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
    // CHECK: %[[GATE:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[QNW:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[KNW:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[POS:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[COS:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[SIN:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[B:.*]] = llvm.extractvalue %[[QM:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[S:.*]] = llvm.extractvalue %[[QM]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
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
    // CHECK: %[[KB:.*]] = llvm.extractvalue %[[KM:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[KS:.*]] = llvm.extractvalue %[[KM]][3, 1]
    // CHECK: %[[KW:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[VB:.*]] = llvm.extractvalue %[[VM:.*]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK: %[[VS:.*]] = llvm.extractvalue %[[VM]][3, 1]
    // CHECK: %[[VW:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[SCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
    // CHECK: %[[EPS:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK-NEXT: %[[NOPAIR:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: llvm.call @wrap_dynamic_sparse_attention(%{{.*}}, %[[Q]], %[[KP]], %[[VP]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[GATE]], %[[QNW]], %[[KNW]], %[[POS]], %[[COS]], %[[SIN]], %{{.*}}, %{{.*}}, %{{.*}}, %[[B]], %[[S]], %[[N]], %[[KVN]], %[[D]], %[[CAP]], %[[K]], %[[PKB]], %[[PKH]], %[[PKD]], %[[PVB]], %[[PVH]], %[[PVC]], %[[PVD]], %[[OKC]], %[[OKD]], %[[OVC]], %[[OVD]], %[[IDXR]], %[[CNTR]], %[[SEQR]], %[[KB]], %[[KS]], %[[KW]], %[[VB]], %[[VS]], %[[VW]], %[[ZERO]], {{(%[0-9]+, ){15}%}}[[SCALE]], %[[EPS]], %[[NOPAIR]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, !llvm.ptr) -> i32
    return
  }

  // A packed query of N + 2 * kvN heads: D = 7168 / 28. Key and value are
  // null; the gate and prologue pass their pointers, dims and attributes.
  func.func @dsa_packed_gate_prologue(
      %ctx: !hip.context,
      %qkv: memref<?x?x7168xf16, 1>,
      %gate: memref<?x?x6144xf16, 1>,
      %qw: memref<256xf16, 1>,
      %kw: memref<256xf16, 1>,
      %pos: memref<3x?x?xi64, 1>,
      %cos: memref<?x32xf16, 1>,
      %sin: memref<?x32xf16, 1>,
      %past_k: memref<?x2x?x?xf16, 1>,
      %past_v: memref<?x2x?x?xf16, 1>,
      %idx: memref<?x2051xi32, 1>,
      %cnt: memref<?xi32, 1>,
      %seqlens: memref<?xi32, 1>,
      %out: memref<?x?x6144xf16, 1>,
      %present_k: memref<?x2x?x?xf16, 1>,
      %present_v: memref<?x2x?x?xf16, 1>) {
    hip.dynamic_sparse_attention(%ctx) ins(%qkv : memref<?x?x7168xf16, 1>)
        cache(%past_k, %past_v : memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        select(%idx, %cnt, %seqlens : memref<?x2051xi32, 1>, memref<?xi32, 1>,
               memref<?xi32, 1>)
        gate(%gate : memref<?x?x6144xf16, 1>)
        prologue(%qw, %kw, %pos, %cos, %sin : memref<256xf16, 1>,
                 memref<256xf16, 1>, memref<3x?x?xi64, 1>, memref<?x32xf16, 1>,
                 memref<?x32xf16, 1>)
        outs(%out, %present_k, %present_v : memref<?x?x6144xf16, 1>,
             memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 2 : i64, scale = 6.250000e-02 : f32,
         norm_epsilon = 9.99999997E-7 : f32, mrope_section = array<i64: 11, 11, 10>,
         mrope_layout = 1 : i64}
    // CHECK-LABEL: llvm.func @dsa_packed_gate_prologue
    // CHECK: %[[QKV:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
    // CHECK-NEXT: %[[NOKEY:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK-NEXT: %[[NOVALUE:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: %[[D:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK: %[[GW:.*]] = llvm.mlir.constant(6144 : i64) : i64
    // CHECK-NEXT: %[[QNL:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-NEXT: %[[KNL:.*]] = llvm.mlir.constant(256 : i64) : i64
    // CHECK-NEXT: %[[SECTIONS:.*]] = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %[[SECT:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK-NEXT: %[[SECH:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK-NEXT: %[[SECW:.*]] = llvm.mlir.constant(10 : i64) : i64
    // CHECK-NEXT: %[[LAYOUT:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: %[[SCALE:.*]] = llvm.mlir.constant(6.250000e-02 : f32) : f32
    // CHECK-NEXT: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
    // CHECK-NEXT: %[[NOPAIR:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK-NEXT: llvm.call @wrap_dynamic_sparse_attention(%{{.*}}, %[[QKV]], %[[NOKEY]], %[[NOVALUE]], {{(%[0-9]+, ){18}%}}[[D]], {{(%[0-9]+, ){24}%}}[[GW]], %[[QNL]], %[[KNL]], %[[SECTIONS]], {{(%[0-9]+, ){6}%}}[[SECT]], %[[SECH]], %[[SECW]], %[[LAYOUT]], %[[SCALE]], %[[EPS]], %[[NOPAIR]])
    return
  }

  // A paired op passes its indexer's past_sequence_length last.
  func.func @dsa_paired(
      %ctx: !hip.context,
      %q: memref<?x?x6144xf16, 1>,
      %k: memref<?x?x512xf16, 1>,
      %v: memref<?x?x512xf16, 1>,
      %past_k: memref<?x2x?x?xf16, 1>,
      %past_v: memref<?x2x?x?xf16, 1>,
      %idx: memref<?x2051xi32, 1>,
      %cnt: memref<?xi32, 1>,
      %seqlens: memref<?xi32, 1>,
      %psl: memref<1xi32, 1>,
      %out: memref<?x?x6144xf16, 1>,
      %present_k: memref<?x2x?x?xf16, 1>,
      %present_v: memref<?x2x?x?xf16, 1>) {
    hip.dynamic_sparse_attention(%ctx) ins(%q : memref<?x?x6144xf16, 1>)
        kv(%k, %v : memref<?x?x512xf16, 1>, memref<?x?x512xf16, 1>)
        cache(%past_k, %past_v : memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        select(%idx, %cnt, %seqlens : memref<?x2051xi32, 1>, memref<?xi32, 1>,
               memref<?xi32, 1>)
        pair(%psl : memref<1xi32, 1>)
        outs(%out, %present_k, %present_v : memref<?x?x6144xf16, 1>,
             memref<?x2x?x?xf16, 1>, memref<?x2x?x?xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 2 : i64, scale = 6.250000e-02 : f32}
    // CHECK-LABEL: llvm.func @dsa_paired
    // CHECK: %[[EPS:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK-NEXT: %[[PSL:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: %[[PAIR:.*]] = llvm.addrspacecast %[[PSL]] : !llvm.ptr<1> to !llvm.ptr
    // CHECK-NEXT: llvm.call @wrap_dynamic_sparse_attention({{.*}}, %[[EPS]], %[[PAIR]]) : ({{.*}}, f32, f32, !llvm.ptr) -> i32
    return
  }
}
