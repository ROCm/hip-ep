// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP memory management operations are correctly lowered to LLVM
// runtime calls when context is passed as a function parameter (real pipeline).
//
// This test validates:
// - hip.alloc → llvm.call @hipMalloc with proper size calculation
// - hip.free → llvm.call @hipFree
// - Context passed as function argument (matches OnnxToHip output)
// - Proper type conversion: !hip.context → !llvm.ptr
// - Memory pointer handling for typed memrefs
// - Dynamic tensor shape handling (?x128)
//
// Architecture: Context managed externally (inference_init/cleanup)
// Expected: LLVM calls to HIP runtime wrapper functions
// ============================================================================

// RUN: morphizen-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test function receives context as parameter (real pipeline pattern)
  // CHECK-LABEL: llvm.func @test_hip_memory_ops
  func.func @test_hip_memory_ops(%ctx: !hip.context, %N: index) {
    // Allocate device memory for a dynamic ?x128 f32 tensor
    // Context is passed to alloc (no create_handle needed)
    // CHECK: llvm.call @hipMalloc(%{{.*}}, %{{.*}}) : (!llvm.ptr, i64) -> i32
    %x = hip.alloc(%ctx, %N) : memref<?x128xf32, 1>

    // Free the device memory
    // Context is passed to free (no destroy_handle needed)
    // CHECK: llvm.call @hipFree(%{{.*}}) : (!llvm.ptr) -> ()
    hip.free(%ctx, %x) : memref<?x128xf32, 1>

    return
  }
}
