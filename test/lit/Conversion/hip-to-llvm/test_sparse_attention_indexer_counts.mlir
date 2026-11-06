// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// The counts init is passed as the last runtime argument.
// CHECK-LABEL: llvm.func @wrap_sparse_attention_indexer(
// CHECK-SAME:      !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr) -> i32
// CHECK-LABEL: llvm.func @sai_counts(
// CHECK-SAME:      %[[CTX:[^:]*]]: !llvm.ptr,
// CHECK:           %[[NOMASK:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %[[PSL_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[PSL:.*]] = llvm.addrspacecast %[[PSL_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[QK_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[QK:.*]] = llvm.addrspacecast %[[QK_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[QW_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[QW:.*]] = llvm.addrspacecast %[[QW_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[KW_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[KW:.*]] = llvm.addrspacecast %[[KW_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[COS_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[COS:.*]] = llvm.addrspacecast %[[COS_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[SIN_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[SIN:.*]] = llvm.addrspacecast %[[SIN_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[PAST_PTR:.*]] = llvm.extractvalue %[[PAST_DESC:.*]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[PAST:.*]] = llvm.addrspacecast %[[PAST_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[SEL_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[SEL:.*]] = llvm.addrspacecast %[[SEL_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[PRESENT_PTR:.*]] = llvm.extractvalue %[[PRESENT_DESC:.*]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[PRESENT:.*]] = llvm.addrspacecast %[[PRESENT_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[S:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[P:.*]] = llvm.extractvalue %[[PAST_DESC]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[PR:.*]] = llvm.extractvalue %[[PRESENT_DESC]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[N:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[D:.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK-NEXT:      %[[R:.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK-NEXT:      %[[X:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:      %[[BATCHED:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[RATIO:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[BUDGET:.*]] = llvm.mlir.constant(2048 : i64) : i64
// CHECK-NEXT:      %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:      %[[PB:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[MB:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[ML:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[CB:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[SB:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[SR:.*]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:      %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
// CHECK-NEXT:      %[[SCALE:.*]] = llvm.mlir.constant(0.0883883461 : f32) : f32
// CHECK-NEXT:      %[[NOIDS:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %[[NOMAP:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[COUNTS_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[COUNTS:.*]] = llvm.addrspacecast %[[COUNTS_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %{{.*}} = llvm.call @wrap_sparse_attention_indexer(%[[CTX]], %[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[NOMASK]], %[[PSL]], %[[PAST]], %[[SEL]], %[[PRESENT]], %[[B]], %[[S]], %[[P]], %[[PR]], %[[N]], %[[D]], %[[R]], %[[X]], %[[BATCHED]], %[[RATIO]], %[[BUDGET]], %[[ELEM]], %[[PB]], %[[MB]], %[[ML]], %[[CB]], %[[SB]], %[[SR]], %[[EPS]], %[[SCALE]], %[[NOIDS]], %[[NOMAP]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[COUNTS]])
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
func.func @sai_counts(
    %ctx: !hip.context,
    %qk: memref<1x1x640xf16, 1>,
    %qw: memref<128xf16, 1>,
    %kw: memref<128xf16, 1>,
    %cos: memref<1x4096x64xf16, 1>,
    %sin: memref<1x4096x64xf16, 1>,
    %past: memref<1x?x128xf16, 1>,
    %psl: memref<1xi32, 1>,
    %sel: memref<1x1x2051xi32, 1>,
    %present: memref<1x?x128xf16, 1>,
    %counts: memref<1x1xi32, 1>) {
  hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          memref<1x1x640xf16, 1>, memref<128xf16, 1>, memref<128xf16, 1>,
          memref<1x4096x64xf16, 1>, memref<1x4096x64xf16, 1>,
          memref<1x?x128xf16, 1>)
      past_sequence_length(%psl : memref<1xi32, 1>)
      outs(%sel, %present : memref<1x1x2051xi32, 1>, memref<1x?x128xf16, 1>)
      counts(%counts : memref<1x1xi32, 1>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32}
  return
}
