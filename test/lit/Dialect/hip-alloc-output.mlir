// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Op-contract tests for hip.alloc_output: parser/printer round-trip (success)
// and verifier rejection of a dynamic-size-operand count that does not match
// the result memref's dynamic-dim count (failure).
//
// NOTE: these are OP-level tests. The op verifier is local -- it checks ONLY
// the dynamic-size operand count against the result memref. It deliberately
// does NOT tie out_idx to a func.return position: the op is valid standalone,
// and an op cannot (and should not) non-locally inspect how/where its result is
// returned. The "out_idx == return-operand position" contract is a property of
// the hip-use-output-allocator PASS, covered in hip-use-output-allocator.mlir
// (@two_outputs). Hence out_idx values below are free-form and intentionally
// need not equal the return index.
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

// --- SUCCESS: static result, no dynamic sizes, round-trips. out_idx = 1 here
//     even though the value is returned at position 0 -- legal at the op level
//     (see header note) and verifies that a non-zero out_idx round-trips. ---
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
