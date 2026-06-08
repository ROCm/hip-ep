// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Op-contract tests for hip.alloc_output: parser/printer round-trip (success)
// and verifier rejection of a dynamic-size-operand count that does not match
// the result memref's dynamic-dim count (failure).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// --- SUCCESS: dynamic result, 2 dynamic sizes, round-trips. ---
// CHECK-LABEL: func.func @roundtrip_dynamic
// CHECK:         hip.alloc_output(%{{.*}}, %{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?x?xf16>
func.func @roundtrip_dynamic(%ctx: !hip.context, %m: index, %n: index) -> memref<?x?xf16> {
  %0 = hip.alloc_output(%ctx, %m, %n) {out_idx = 0 : i64} : memref<?x?xf16>
  return %0 : memref<?x?xf16>
}

// -----

// --- SUCCESS: static result, no dynamic sizes, round-trips. ---
// CHECK-LABEL: func.func @roundtrip_static
// CHECK:         hip.alloc_output(%{{.*}}) {out_idx = 1 : i64} : memref<4x8xf16>
func.func @roundtrip_static(%ctx: !hip.context) -> memref<4x8xf16> {
  %0 = hip.alloc_output(%ctx) {out_idx = 1 : i64} : memref<4x8xf16>
  return %0 : memref<4x8xf16>
}

// -----

// --- FAIL: 2 dynamic dims but only 1 dynamic-size operand. ---
func.func @too_few_sizes(%ctx: !hip.context, %m: index) -> memref<?x?xf16> {
  // expected-error @below {{expected 2 dynamic size operand(s), got 1}}
  %0 = hip.alloc_output(%ctx, %m) {out_idx = 0 : i64} : memref<?x?xf16>
  return %0 : memref<?x?xf16>
}

// -----

// --- FAIL: 1 dynamic dim but 2 dynamic-size operands. ---
func.func @too_many_sizes(%ctx: !hip.context, %m: index, %n: index) -> memref<?xf16> {
  // expected-error @below {{expected 1 dynamic size operand(s), got 2}}
  %0 = hip.alloc_output(%ctx, %m, %n) {out_idx = 0 : i64} : memref<?xf16>
  return %0 : memref<?xf16>
}
