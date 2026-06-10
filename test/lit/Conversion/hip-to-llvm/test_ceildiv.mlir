// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify arith.ceildivsi / arith.floordivsi survive the HIP->LLVM partial
// conversion by being EXPANDED into primitive arith ops (which the ArithToLLVM
// patterns then lower), rather than being left untouched.
//
// ArithToLLVM has no direct pattern for ceildivsi / floordivsi / ceildivui;
// without arith::populateCeilFloorDivExpandOpsPatterns in the conversion, a
// stray ceildivsi (e.g. from dynamic-shape index arithmetic in an outlined
// loop body) survives applyPartialConversion and aborts MLIR->LLVM translation
// with "missing LLVMTranslationDialectInterface ... for op: arith.ceildivsi".
//
// Expected: the function lowers to llvm.func and NO arith.{ceildivsi,
// floordivsi} op remains in the output.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @ceildiv_floordiv_i64
  // CHECK-NOT: arith.ceildivsi
  // CHECK-NOT: arith.floordivsi
  func.func @ceildiv_floordiv_i64(%a: i64, %b: i64) -> i64 {
    %0 = arith.ceildivsi %a, %b : i64
    %1 = arith.floordivsi %0, %b : i64
    func.return %1 : i64
  }
}
