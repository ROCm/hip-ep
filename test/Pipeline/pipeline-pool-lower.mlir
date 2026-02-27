// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: hip-pool-allocs -> hip-lower-allocs
//
// Verifies that pooling produces a single memref.alloc (i8 pool) which
// hip-lower-allocs then converts to a single hip.alloc + hip.free.
//===----------------------------------------------------------------------===//

// RUN: %hip-mlir-opt --hip-pool-allocs --hip-lower-allocs %s | %FileCheck %s

// ===== Static model: pool + lower =====
//
// Two memref<8x8xf32> allocs (256 bytes each) become:
//   1. hip-pool-allocs: memref<512xi8> pool + two memref.view
//   2. hip-lower-allocs: hip.alloc for the pool, two memref.view
//
// The pool is NOT freed because a view of it (%alloc1) is returned.
//
// CHECK-LABEL: func.func @static_pool_then_lower
// CHECK:         %[[H:.*]] = hip.create_handle()
// CHECK:         %[[POOL:.*]] = hip.alloc(%[[H]]) : memref<512xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<512xi8> to memref<8x8xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<512xi8> to memref<8x8xf32>
// CHECK:         hip.miopen.softmax
// CHECK-NOT:     hip.free
// CHECK:         hip.destroy_handle(%[[H]])
// CHECK:         return
func.func @static_pool_then_lower(
    %a: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>) -> memref<8x8xf32> {
  %handle = hip.create_handle() : !hip.handle
  %alloc0 = memref.alloc() : memref<8x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<8x8xf32, strided<[?, ?], offset: ?>>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<8x8xf32>)
  %alloc1 = memref.alloc() : memref<8x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<8x8xf32>) outs(%alloc1 : memref<8x8xf32>)
  hip.destroy_handle(%handle) : !hip.handle
  return %alloc1 : memref<8x8xf32>
}

// ===== Dynamic model: pool + lower =====
//
// Two memref<?x8xf32> allocs (same %n) become:
//   1. hip-pool-allocs: memref<?xi8> pool + two memref.view
//   2. hip-lower-allocs: hip.alloc for the pool, two memref.view
//
// The pool is NOT freed because a view of it (%alloc1) is returned.
//
// CHECK-LABEL: func.func @dynamic_pool_then_lower
// CHECK:         %[[H:.*]] = hip.create_handle()
// CHECK:         %[[POOL:.*]] = hip.alloc(%[[H]]{{.*}}) : memref<?xi8>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         hip.hipblaslt.matmul
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<?x8xf32>
// CHECK:         hip.miopen.softmax
// CHECK-NOT:     hip.free
// CHECK:         hip.destroy_handle(%[[H]])
// CHECK:         return
func.func @dynamic_pool_then_lower(
    %a: memref<?x8xf32>,
    %b: memref<8x8xf32, strided<[?, ?], offset: ?>>,
    %n: index) -> memref<?x8xf32> {
  %handle = hip.create_handle() : !hip.handle
  %alloc0 = memref.alloc(%n) : memref<?x8xf32>
  hip.hipblaslt.matmul(%handle) ins(%a, %b : memref<?x8xf32>, memref<8x8xf32, strided<[?, ?], offset: ?>>) outs(%alloc0 : memref<?x8xf32>)
  %alloc1 = memref.alloc(%n) : memref<?x8xf32>
  hip.miopen.softmax(%handle) ins(%alloc0 : memref<?x8xf32>) outs(%alloc1 : memref<?x8xf32>)
  hip.destroy_handle(%handle) : !hip.handle
  return %alloc1 : memref<?x8xf32>
}
