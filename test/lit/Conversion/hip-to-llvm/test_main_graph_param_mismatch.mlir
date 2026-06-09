// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: convert-hip-to-llvm's transformMainFunction rejects a @main_graph whose
// post-lowering parameter count does not match the ABI selected by the
// `hipdnn.output_allocator` module attribute.
//
// Mode is chosen by the attribute (set by hip-set-output-allocator-attr), NOT by
// param count. The expected count for the chosen mode is:
//   expectedAllocator = 1 + sum_inputs(3 + 2*rank)            (context + inputs)
//   expectedClassic   = expectedAllocator + sum_outputs(3 + 2*rank)
//
// Here: 1 input of rank 2 => expectedAllocator = 1 + (3 + 4) = 8;
//       1 output of rank 2 => expectedClassic = 8 + (3 + 4) = 15.
// @main_graph below has 5 params (neither), so the pass must emit a precise,
// MODE-SPECIFIC mismatch diagnostic and fail rather than silently mis-wrapping.
// Two modules (split-input-file) pin both diagnostics: classic (no attribute)
// and allocator (attribute present).
// ============================================================================

// RUN: not hip-mlir-opt --convert-hip-to-llvm -split-input-file %s 2>&1 | FileCheck %s

// --- Classic (no hipdnn.output_allocator attr): expected = expectedClassic. ---
// CHECK: @main_graph parameter count mismatch: expected 15 (classic), got 5
module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 1, 4>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 1, 4>]
} {
  // Already LLVM dialect: applyPartialConversion is a no-op, so transformMain
  // sees this signature verbatim. 5 params != 15 (classic).
  llvm.func @main_graph(%a: !llvm.ptr, %b: !llvm.ptr, %c: i64, %d: i64, %e: i64) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}

// -----

// --- Allocator (hipdnn.output_allocator set): expected = expectedAllocator. ---
// CHECK: @main_graph parameter count mismatch: expected 8 (allocator), got 5
module attributes {
  hipdnn.output_allocator,
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: 1, 4>],
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: 1, 4>]
} {
  // Same 5-param signature; allocator mode expects 8 (context + 1 rank-2 input).
  llvm.func @main_graph(%a: !llvm.ptr, %b: !llvm.ptr, %c: i64, %d: i64, %e: i64) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
