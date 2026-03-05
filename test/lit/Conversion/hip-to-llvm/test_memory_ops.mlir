// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.get_pool is correctly lowered to a call to hipdnn_ep_get_pool_base.
//
// This test validates:
// - hip.get_pool(%ctx) → llvm.call @hipdnn_ep_get_pool_base(state)
// - Returned GPU pointer is wrapped in a memref descriptor
// - Proper type conversion: !hip.context → !llvm.ptr
//
// Architecture: Pool base pointer is obtained from the runtime state and
// wrapped as a memref<?xi8, 1>. Slices are taken via memref.view at
// static offsets assigned by the memory-pooling pass.
// ============================================================================

// RUN: udna-opt %s --convert-hip-to-llvm | FileCheck %s

module attributes {"hipdnn.pool_size" = 65536 : i64} {
  // CHECK-LABEL: llvm.func @test_get_pool
  func.func @test_get_pool(%ctx: !hip.context) -> memref<?xi8, 1> {
    // hip.get_pool lowers to hipdnn_ep_get_pool_base runtime call
    // CHECK: llvm.call @hipdnn_ep_get_pool_base({{.*}}) : (!llvm.ptr) -> !llvm.ptr
    %pool = hip.get_pool(%ctx) : memref<?xi8, 1>
    return %pool : memref<?xi8, 1>
  }
}
