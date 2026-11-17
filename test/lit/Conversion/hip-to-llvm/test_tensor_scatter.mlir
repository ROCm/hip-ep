// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.tensor_scatter lowers to wrap_tensor_scatter(state, past, update,
// write_indices, output, elem_bytes, batch, outer, max_seq, seq, inner,
// circular): outer is the product of the dims between batch and axis, inner
// the product of the dims after axis, write_indices null when absent.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // ===== KV cache [B, heads, max_seq, head]: axis 2, dynamic update length =====
  func.func @kv_cache(%ctx: !hip.context, %past: memref<1x4x64x8xf16, 1>,
                      %update: memref<1x4x?x8xf16, 1>, %wi: memref<1xi64, 1>,
                      %out: memref<1x4x64x8xf16, 1>) {
    hip.tensor_scatter(%ctx) ins(%past, %update : memref<1x4x64x8xf16, 1>,
                                 memref<1x4x?x8xf16, 1>)
        write_indices(%wi : memref<1xi64, 1>)
        outs(%out : memref<1x4x64x8xf16, 1>) {axis = 2 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @kv_cache
  // CHECK: %[[EB:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[B:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[O:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
  // CHECK: %[[M:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK: %[[S:.*]] = llvm.extractvalue %{{.*}}[3, 2]
  // CHECK: %[[I:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
  // CHECK: %[[C:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @wrap_tensor_scatter(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[EB]], %[[B]], %[[O]], %[[M]], %[[S]], %[[I]], %[[C]])
  // CHECK-SAME: (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32

  // ===== rank 3, axis 1, no write_indices, circular =====
  func.func @circular(%ctx: !hip.context, %past: memref<2x16x8xf32, 1>,
                      %update: memref<2x3x8xf32, 1>,
                      %out: memref<2x16x8xf32, 1>) {
    hip.tensor_scatter(%ctx) ins(%past, %update : memref<2x16x8xf32, 1>,
                                 memref<2x3x8xf32, 1>)
        outs(%out : memref<2x16x8xf32, 1>) {axis = 1 : i64, circular = true}
    return
  }

  // CHECK-LABEL: llvm.func @circular
  // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[EB4:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK-NEXT: %[[B2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-NEXT: %[[O1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: %[[M16:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-NEXT: %[[S3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[I8:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
  // CHECK-NEXT: %[[CIRC:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-NEXT: llvm.call @wrap_tensor_scatter(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[NULL]], %{{[^,]*}}, %[[EB4]], %[[B2]], %[[O1]], %[[M16]], %[[S3]], %[[I8]], %[[CIRC]])
}
