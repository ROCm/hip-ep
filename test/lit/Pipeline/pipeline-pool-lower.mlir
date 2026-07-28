// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: hip-pool-allocs -> convert-hip-to-llvm
//
// Verifies that pooling + LLVM lowering produces llvm.call
// @hipdnn_ep_get_pool_base for the pool and llvm.call @hip_* for compute ops.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-allocs --convert-hip-to-llvm %s | FileCheck %s

// ===== Static model: pool + LLVM lowering =====
//
// Two memref<8x8xf32> allocs (256 bytes each), pool = 512 bytes.
// hip.get_pool -> llvm.call @hipdnn_ep_get_pool_base(%arg0)
// Pool size 512 appears as llvm.mlir.constant, offset 256 for second view.
//
// CHECK-LABEL: llvm.func @static_pool_to_llvm
// CHECK-SAME:    (%[[CTX:[a-z0-9]+]]: !llvm.ptr,
// CHECK:         %[[POOL_SIZE:.*]] = llvm.mlir.constant(512 : index) : i64
// CHECK:         %[[DOM:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @hipdnn_ep_get_pool_base(%[[CTX]], %[[DOM]], %[[POOL_SIZE]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK:         llvm.mlir.constant(256 : index) : i64
// CHECK:         llvm.call @wrap_hipblasLtMatmul_v2(%[[CTX]],
// CHECK:         llvm.call @hip_miopen_softmax(%[[CTX]],
// CHECK:         llvm.return
func.func @static_pool_to_llvm(
    %ctx: !hip.context,
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  return %alloc1 : memref<8x8xf32>
}

// ===== Dynamic model: pool + LLVM lowering =====
//
// Two memref<?x8xf32> allocs (same %n). Byte size = n * 32 (f32 * 8).
// Pool size is computed at runtime: alignUp(n*32, 256) * 2.
//
// CHECK-LABEL: llvm.func @dynamic_pool_to_llvm
// CHECK-SAME:    (%[[CTX2:[a-z0-9]+]]: !llvm.ptr,
// CHECK:         %[[C32:[a-z0-9_]+]] = llvm.mlir.constant(32 : index) : i64
// CHECK:         llvm.mul %arg15, %[[C32]] : i64
// CHECK:         %[[DOM2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @hipdnn_ep_get_pool_base(%[[CTX2]], %[[DOM2]], %{{[0-9]+}}) : (!llvm.ptr, i32, i64) -> !llvm.ptr
// CHECK:         llvm.call @wrap_hipblasLtMatmul_v2(%[[CTX2]],
// CHECK:         llvm.call @hip_miopen_softmax(%[[CTX2]],
// CHECK:         llvm.return
func.func @dynamic_pool_to_llvm(
    %ctx: !hip.context,
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x8xf32> {
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.matmul(%ctx) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%ctx) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  return %alloc1 : memref<?x8xf32>
}
