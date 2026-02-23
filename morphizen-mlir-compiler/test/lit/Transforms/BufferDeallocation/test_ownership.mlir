// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify BufferDeallocation pass correctly distinguishes between
// function-owned and caller-owned buffers.
//
// This test validates:
// - Deallocation of function-owned buffers (allocated via hip.alloc)
// - NO deallocation of caller-owned buffers (function arguments)
// - Ownership tracking across buffer operations
// - Correct use of CHECK-NOT to verify absence of unwanted frees
//
// Ownership rules:
// - Function arguments: Caller owns, function must not free
// - Local allocations: Function owns, function must free before return
//
// Expected: hip.free for temp only, NOT for input (function argument)
// ============================================================================

// RUN: morphizen-opt %s --hip-buffer-deallocation | FileCheck %s

module {
  func.func @test_ownership(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>) -> i32 {
    // CHECK-LABEL: func.func @test_ownership

    // Allocate temp buffer (function owns this)
    // CHECK: %[[TEMP:.*]] = hip.alloc(%{{.*}})
    %temp = hip.alloc(%ctx) : memref<1x3x224x224xf32, 1>

    // Copy from input (caller-owned) to temp (function-owned)
    memref.copy %input, %temp : memref<1x3x224x224xf32, 1> to memref<1x3x224x224xf32, 1>

    // BufferDeallocation should insert: hip.free(%ctx, %temp)
    // BufferDeallocation should NOT insert: hip.free(%ctx, %input)
    // CHECK: hip.free(%{{.*}}, %[[TEMP]])
    // CHECK-NOT: hip.free(%{{.*}}, %{{.*input.*}})

    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
