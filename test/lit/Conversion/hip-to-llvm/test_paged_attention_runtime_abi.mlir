// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Check all 36 parameter types, including the two independently sized extents.
// CHECK: llvm.func @wrap_paged_attention_v2(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32) -> i32
// CHECK-LABEL: llvm.func @paged_runtime_extents
// Tie extent extraction to the same descriptors used for each data pointer.
// CHECK: llvm.extractvalue %[[Q_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: llvm.extractvalue %[[K_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: llvm.extractvalue %[[V_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: llvm.extractvalue %[[KP_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: %[[VP_DATA:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[VP_DESC:[a-zA-Z0-9_]+]][1]
// CHECK-NEXT: %[[VP_PTR:[a-zA-Z0-9_]+]] = llvm.addrspacecast %[[VP_DATA]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: llvm.extractvalue %[[CUM_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: llvm.extractvalue %[[PAST_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: llvm.extractvalue %[[TABLE_DESC:[a-zA-Z0-9_]+]][1]
// CHECK: %[[OUT_DATA:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OUT_DESC:[a-zA-Z0-9_]+]][1]
// CHECK-NEXT: %[[OUT_PTR:[a-zA-Z0-9_]+]] = llvm.addrspacecast %[[OUT_DATA]] : !llvm.ptr<1> to !llvm.ptr
// CHECK: %[[Q_T:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[Q_DESC]][3, 0]
// CHECK-NEXT: %[[K_T:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[K_DESC]][3, 0]
// CHECK-NEXT: %[[V_T:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[V_DESC]][3, 0]
// CHECK-NEXT: %[[OUT_T:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OUT_DESC]][3, 0]
// CHECK-NEXT: %[[BATCH:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[PAST_DESC]][3, 0]
// CHECK-NEXT: %[[K_PAGES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[KP_DESC]][3, 0]
// CHECK-NEXT: %[[V_PAGES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[VP_DESC]][3, 0]
// CHECK-NEXT: %[[K_PAGE_SIZE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[KP_DESC]][3, 1]
// CHECK-NEXT: %[[V_PAGE_SIZE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[VP_DESC]][3, 1]
// The runtime must receive output.dim0 after Q/K/V token counts, and V.dim1
// after K.dim1. Checking the full call type also rejects the previous 37-arg ABI.
// CHECK: llvm.call @wrap_paged_attention_v2(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[VP_PTR]], %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %[[OUT_PTR]], %[[Q_T]], %[[K_T]], %[[V_T]], %[[OUT_T]], %[[BATCH]], %[[K_PAGES]], %[[V_PAGES]], %[[K_PAGE_SIZE]], %[[V_PAGE_SIZE]], %{{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32) -> i32

// Independent dynamic descriptors make accidental reuse of Q's token count or
// K's page size visible in the runtime ABI. This only compiles Hip IR.
module {
  func.func @paged_runtime_extents(
      %ctx: !hip.context,
      %q: memref<?x6144xf16, 1>,
      %k: memref<?x1024xf16, 1>,
      %v: memref<?x1024xf16, 1>,
      %kp: memref<?x?x4x256xf16, 1>,
      %vp: memref<?x?x4x256xf16, 1>,
      %cum: memref<?xi32, 1>,
      %past: memref<?xi32, 1>,
      %table: memref<?x?xi32, 1>,
      %out: memref<?x6144xf16, 1>) {
    hip.paged_attention(%ctx)
        ins(%q, %k, %v, %cum, %past, %table :
            memref<?x6144xf16, 1>, memref<?x1024xf16, 1>,
            memref<?x1024xf16, 1>, memref<?xi32, 1>, memref<?xi32, 1>,
            memref<?x?xi32, 1>)
        outs(%out, %kp, %vp : memref<?x6144xf16, 1>,
            memref<?x?x4x256xf16, 1>, memref<?x?x4x256xf16, 1>)
        {num_heads = 24 : i64, kv_num_heads = 4 : i64,
         hip.op_state_slot = 0 : i32}
    return
  }
}
