// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-pool-allocs %s | FileCheck %s
//
// Both functions use local domain zero. Their deterministic module-site IDs
// must differ so an outlined helper cannot share its caller's live pool.

// CHECK-LABEL: func.func @caller
// CHECK: hip.get_pool({{.*}}) : memref<?xi8>
func.func @caller(%ctx: !hip.context, %in: memref<8xf32>) -> memref<8xf32> {
  %tmp = memref.alloc() : memref<8xf32>
  hip.miopen.softmax(%ctx) ins(%in : memref<8xf32>) outs(%tmp : memref<8xf32>)
  return %tmp : memref<8xf32>
}

// CHECK-LABEL: func.func @outlined_helper
// CHECK: hip.get_pool({{.*}}) {site_id = 1 : i64} : memref<?xi8>
func.func @outlined_helper(
    %ctx: !hip.context, %in: memref<16xf32>) -> memref<16xf32> {
  %tmp = memref.alloc() : memref<16xf32>
  hip.miopen.softmax(%ctx) ins(%in : memref<16xf32>) outs(%tmp : memref<16xf32>)
  return %tmp : memref<16xf32>
}
