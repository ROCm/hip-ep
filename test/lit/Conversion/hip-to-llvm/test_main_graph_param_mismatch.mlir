// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: convert-hip-to-llvm's transformMainFunction rejects a @main_graph whose
// post-lowering parameter count does not match the output-allocator ABI.
//
// The expected count is context + inputs only (outputs are allocated in-graph):
//   expected = 1 + sum_inputs(3 + 2*rank)
//
// Here: 1 input of rank 2 => expected = 1 + (3 + 4) = 8.
// @main_graph below has 5 params, so the pass must emit a precise mismatch
// diagnostic and fail rather than silently mis-wrapping.
// ============================================================================

// RUN: not hip-mlir-opt --convert-hip-to-llvm %s 2>&1 | FileCheck %s

// CHECK: @main_graph parameter count mismatch: expected 8, got 5
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 1, 4>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 1, 4>]
} {
  // Already LLVM dialect: applyPartialConversion is a no-op, so transformMain
  // sees this signature verbatim. 5 params != 8 (context + 1 rank-2 input).
  llvm.func @main_graph(%a: !llvm.ptr, %b: !llvm.ptr, %c: i64, %d: i64, %e: i64) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
