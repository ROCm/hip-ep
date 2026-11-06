// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Base tables [16, 4] gathered through [3, 1, X] position ids. The rotary
// width is 8, X comes from the ids, and both indexers share one section map.
// CHECK-LABEL: llvm.func @wrap_sparse_attention_indexer(
// CHECK-SAME:      !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    llvm.mlir.global internal constant @__hip_sai_rotary_sections_00010200("\00\01\02\00") {addr_space = 0 : i32}
// CHECK-NOT:     llvm.mlir.global
// CHECK-LABEL: llvm.func @sai_rotary_ids(
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
// CHECK-NEXT:      %[[COS_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[COS:.*]] = llvm.addrspacecast %[[COS_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[SIN_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:      %[[SIN:.*]] = llvm.addrspacecast %[[SIN_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[PAST_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[PAST:.*]] = llvm.addrspacecast %[[PAST_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[SEL_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[SEL:.*]] = llvm.addrspacecast %[[SEL_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[PRESENT_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[PRESENT:.*]] = llvm.addrspacecast %[[PRESENT_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[S:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[P:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:      %[[PR:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:      %[[N:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[D:.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK-NEXT:      %[[R:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:      %[[Y:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:      %[[BATCHED:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[RATIO:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:      %[[BUDGET:.*]] = llvm.mlir.constant(2048 : i64) : i64
// CHECK-NEXT:      %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:      %[[PB:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[MB:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[ML:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[CB:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[SB:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:      %[[SR:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-NEXT:      %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
// CHECK-NEXT:      %[[SCALE:.*]] = llvm.mlir.constant(0.0883883461 : f32) : f32
// CHECK-NEXT:      %[[IDS_PTR:.*]] = llvm.extractvalue %[[IDS_DESC:.*]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[IDS:.*]] = llvm.addrspacecast %[[IDS_PTR]] : !llvm.ptr<1> to !llvm.ptr
// CHECK-NEXT:      %[[MAP_GLOBAL:.*]] = llvm.mlir.addressof @__hip_sai_rotary_sections_00010200 : !llvm.ptr
// CHECK-NEXT:      %[[MAP:.*]] = llvm.getelementptr %[[MAP_GLOBAL]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK-NEXT:      %[[M:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:      %[[IDS_B:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[IDS_X:.*]] = llvm.extractvalue %[[IDS_DESC]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[NOCOUNTS:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %{{.*}} = llvm.call @wrap_sparse_attention_indexer(%[[CTX]], %[[QK]], %[[QW]], %[[KW]], %[[COS]], %[[SIN]], %[[NOMASK]], %[[PSL]], %[[PAST]], %[[SEL]], %[[PRESENT]], %[[B]], %[[S]], %[[P]], %[[PR]], %[[N]], %[[D]], %[[R]], %[[Y]], %[[BATCHED]], %[[RATIO]], %[[BUDGET]], %[[ELEM]], %[[PB]], %[[MB]], %[[ML]], %[[CB]], %[[SB]], %[[SR]], %[[EPS]], %[[SCALE]], %[[IDS]], %[[MAP]], %[[M]], %[[IDS_B]], %[[IDS_X]], %[[NOCOUNTS]])
// CHECK:           %[[MAP2:.*]] = llvm.mlir.addressof @__hip_sai_rotary_sections_00010200 : !llvm.ptr
// CHECK-NEXT:      %[[M2:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:      %[[IDS_B2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:      %[[IDS_X2:.*]] = llvm.extractvalue %[[IDS_DESC]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:      %[[NOCOUNTS2:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:      %{{.*}} = llvm.call @wrap_sparse_attention_indexer(%[[CTX]], {{.*}}, %[[MAP2]], %[[M2]], %[[IDS_B2]], %[[IDS_X2]], %[[NOCOUNTS2]])
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
func.func @sai_rotary_ids(
    %ctx: !hip.context,
    %qk: memref<1x1x640xf16, 1>,
    %qw: memref<128xf16, 1>,
    %kw: memref<128xf16, 1>,
    %cos: memref<16x4xf16, 1>,
    %sin: memref<16x4xf16, 1>,
    %past: memref<1x8x128xf16, 1>,
    %psl: memref<1xi32, 1>,
    %ids: memref<3x1x?xi64, 1>,
    %sel: memref<1x1x2051xi32, 1>,
    %present: memref<1x8x128xf16, 1>) {
  hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          memref<1x1x640xf16, 1>, memref<128xf16, 1>, memref<128xf16, 1>,
          memref<16x4xf16, 1>, memref<16x4xf16, 1>, memref<1x8x128xf16, 1>)
      past_sequence_length(%psl : memref<1xi32, 1>)
      rotary_position_ids(%ids : memref<3x1x?xi64, 1>)
      outs(%sel, %present : memref<1x1x2051xi32, 1>, memref<1x8x128xf16, 1>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
  hip.sparse_attention_indexer(%ctx)
      ins(%qk, %qw, %kw, %cos, %sin, %past :
          memref<1x1x640xf16, 1>, memref<128xf16, 1>, memref<128xf16, 1>,
          memref<16x4xf16, 1>, memref<16x4xf16, 1>, memref<1x8x128xf16, 1>)
      past_sequence_length(%psl : memref<1xi32, 1>)
      rotary_position_ids(%ids : memref<3x1x?xi64, 1>)
      outs(%sel, %present : memref<1x1x2051xi32, 1>, memref<1x8x128xf16, 1>)
      {compress_ratio = 4 : i64, token_budget = 2048 : i64,
       epsilon = 9.99999997E-7 : f32, scale = 0.0883883461 : f32,
       rotary_sections = array<i32: 0, 1, 2, 0>}
  return
}
