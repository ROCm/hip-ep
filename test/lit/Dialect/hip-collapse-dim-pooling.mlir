// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// A contiguous copy of a strided QKV slice must use the explicit sequence
// extent, rather than query a late view of the QKV allocation.
// RUN: hip-mlir-opt --hip-hoist-alloc-size-arith --hip-pool-allocs \
// RUN:   --verify-each %s | FileCheck %s --check-prefix=BASE
// RUN: hip-mlir-opt --hip-resolve-memref-dims --cse --canonicalize \
// RUN:   --hip-hoist-alloc-size-arith --hip-pool-allocs --verify-each %s \
// RUN:   | FileCheck %s --check-prefix=FIXED --implicit-check-not=hip.get_pool

// BASE-LABEL: func.func @strided_copy(
// BASE-COUNT-2: hip.get_pool
// BASE-NOT: hip.get_pool

// FIXED-LABEL: func.func @strided_copy(
// FIXED-SAME: %[[CTX:.*]]: !hip.context, %[[SRC:.*]]: memref<?x96xf32>, %[[N:.*]]: index,
// FIXED: %[[POOL:.*]] = hip.get_pool
// FIXED: memref.expand_shape
// FIXED: memref.subview
// FIXED: %[[VIEW:.*]] = memref.collapse_shape
// FIXED-NOT: memref.dim
// FIXED-NOT: hip.get_pool
// FIXED: %[[COPY:.*]] = memref.view %[[POOL]][{{.*}}][%[[N]]] : memref<?xi8> to memref<?x4x8xf32>
// FIXED: memref.copy %[[VIEW]], %[[COPY]]
// FIXED-NOT: hip.get_pool
// FIXED: return
func.func @strided_copy(%ctx: !hip.context, %src: memref<?x96xf32>,
                       %n: index, %dst: memref<?x4x8xf32>) {
  %c0 = arith.constant 0 : index
  %qkv = memref.alloc(%n) : memref<?x96xf32>
  memref.copy %src, %qkv : memref<?x96xf32> to memref<?x96xf32>
  %expanded = memref.expand_shape %qkv [[0], [1, 2, 3]]
      output_shape [%n, 3, 4, 8] : memref<?x96xf32> into memref<?x3x4x8xf32>
  %slice = memref.subview %expanded[0, 1, 0, 0] [%n, 1, 4, 8] [1, 1, 1, 1]
      : memref<?x3x4x8xf32> to memref<?x1x4x8xf32, strided<[96, 32, 8, 1], offset: 32>>
  %collapsed = memref.collapse_shape %slice [[0], [1, 2], [3]]
      : memref<?x1x4x8xf32, strided<[96, 32, 8, 1], offset: 32>>
      into memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>
  %late_n = memref.dim %collapsed, %c0
      : memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>
  %copy = memref.alloc(%late_n) : memref<?x4x8xf32>
  memref.copy %collapsed, %copy
      : memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>> to memref<?x4x8xf32>
  memref.copy %copy, %dst : memref<?x4x8xf32> to memref<?x4x8xf32>
  return
}
