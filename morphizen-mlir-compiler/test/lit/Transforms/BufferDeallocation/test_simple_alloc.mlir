// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify BufferDeallocation pass automatically inserts hip.free operations
// for locally allocated buffers.
//
// This test validates:
// - Automatic hip.free insertion after last use of hip.alloc
// - Multiple buffer handling (buf1, buf2)
// - Reverse-order deallocation (LIFO: buf2 freed before buf1)
// - No deallocation of function arguments (only local allocations)
//
// BufferDeallocation implements automatic memory management:
// - Tracks ownership of allocated buffers
// - Inserts deallocation before function return
// - Follows RAII-style lifetime management
//
// Expected: hip.free calls in reverse allocation order before return
// ============================================================================

// RUN: morphizen-opt %s --hip-buffer-deallocation | FileCheck %s

module {
  func.func @simple_alloc(%ctx: !hip.context) -> i32 {
    // CHECK-LABEL: func.func @simple_alloc
    // CHECK-SAME: %[[CTX:.*]]: !hip.context

    // Allocate two buffers
    // CHECK: %[[BUF1:.*]] = hip.alloc(%[[CTX]])
    %buf1 = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
    // CHECK: %[[BUF2:.*]] = hip.alloc(%[[CTX]])
    %buf2 = hip.alloc(%ctx) : memref<1x64x112x112xf32, 1>

    %c0 = arith.constant 0 : i32
    // CHECK: arith.constant 0

    // After BufferDeallocation, hip.free should be inserted before return in LIFO order:
    // CHECK-NEXT: hip.free(%[[CTX]], %[[BUF2]])
    // CHECK-NEXT: hip.free(%[[CTX]], %[[BUF1]])
    // CHECK-NEXT: return
    return %c0 : i32
  }
}
