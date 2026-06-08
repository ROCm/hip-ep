// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: convert-hip-to-llvm's transformMainFunction rejects a @main_graph whose
// post-lowering parameter count matches NEITHER the classic nor the allocator
// ABI.
//
// transformMainFunction auto-detects mode by counting unpacked memref params
// (each memref => 3 + 2*rank LLVM params; the context adds 1):
//   expectedAllocator = 1 + sum_inputs(3 + 2*rank)
//   expectedClassic   = expectedAllocator + sum_outputs(3 + 2*rank)
//
// Here: 1 input of rank 2 => expectedAllocator = 1 + (3 + 4) = 8;
//       1 output of rank 2 => expectedClassic = 8 + (3 + 4) = 15.
// @main_graph below has 5 params (neither), so the pass must emit a precise
// mismatch diagnostic and fail rather than silently mis-wrapping the function.
// ============================================================================

// RUN: not hip-mlir-opt --convert-hip-to-llvm %s 2>&1 | FileCheck %s

// CHECK: @main_graph parameter count mismatch: expected 15 (classic) or 8 (allocator), got 5

module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 1, 4>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 1, 4>]
} {
  // Already LLVM dialect: applyPartialConversion is a no-op, so transformMain
  // sees this signature verbatim. 5 params != 8 (allocator) and != 15 (classic).
  llvm.func @main_graph(%a: !llvm.ptr, %b: !llvm.ptr, %c: i64, %d: i64, %e: i64) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
