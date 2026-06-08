// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-use-output-allocator.
//
// Verifies that the pass rewrites graph-output memref.alloc ops (values
// returned by func.return) into hip.alloc_output, reusing the alloc's dynamic
// sizes and setting out_idx to the return position, while leaving intermediates,
// passthrough outputs, and context-less functions untouched.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-use-output-allocator %s 2>&1 | FileCheck %s

// --- Dynamic-shape output: replaced, reusing the alloc's %M, %N. ---
// CHECK-LABEL: func.func @dynamic_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[M:.*]]: index, %[[N:.*]]: index)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]], %[[M]], %[[N]]) {out_idx = 0 : i64} : memref<?x?xf16>
// CHECK:         return %[[OUT]]
func.func @dynamic_output(%ctx: !hip.context, %M: index, %N: index) -> memref<?x?xf16> {
  %out = memref.alloc(%M, %N) : memref<?x?xf16>
  return %out : memref<?x?xf16>
}

// --- Intermediate alloc (not returned) stays a memref.alloc; only the
//     returned output is rewritten. ---
// CHECK-LABEL: func.func @intermediate_stays
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[M:.*]]: index, %[[N:.*]]: index)
// CHECK:         %[[T:.*]] = memref.alloc(%[[M]], %[[N]]) : memref<?x?xf16>
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]], %[[M]], %[[N]]) {out_idx = 0 : i64} : memref<?x?xf16>
// CHECK:         memref.copy %[[T]], %[[OUT]]
// CHECK:         return %[[OUT]]
func.func @intermediate_stays(%ctx: !hip.context, %M: index, %N: index) -> memref<?x?xf16> {
  %t = memref.alloc(%M, %N) : memref<?x?xf16>
  %out = memref.alloc(%M, %N) : memref<?x?xf16>
  memref.copy %t, %out : memref<?x?xf16> to memref<?x?xf16>
  return %out : memref<?x?xf16>
}

// --- Two outputs get out_idx 0 and 1 by return position. ---
// CHECK-LABEL: func.func @two_outputs
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[M:.*]]: index)
// CHECK:         hip.alloc_output(%[[CTX]], %[[M]]) {out_idx = 0 : i64} : memref<?xf16>
// CHECK:         hip.alloc_output(%[[CTX]]) {out_idx = 1 : i64} : memref<4xf16>
func.func @two_outputs(%ctx: !hip.context, %M: index) -> (memref<?xf16>, memref<4xf16>) {
  %a = memref.alloc(%M) : memref<?xf16>
  %b = memref.alloc() : memref<4xf16>
  return %a, %b : memref<?xf16>, memref<4xf16>
}

// --- Static-shape output: no dynamic-size operands. ---
// CHECK-LABEL: func.func @static_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<4x8xf16>
func.func @static_output(%ctx: !hip.context) -> memref<4x8xf16> {
  %out = memref.alloc() : memref<4x8xf16>
  return %out : memref<4x8xf16>
}

// --- Passthrough output (return a memref block-arg): left unchanged. ---
// CHECK-LABEL: func.func @passthrough
// CHECK-NOT:     hip.alloc_output
// CHECK:         return %{{.*}} : memref<?xf16>
func.func @passthrough(%ctx: !hip.context, %x: memref<?xf16>) -> memref<?xf16> {
  return %x : memref<?xf16>
}

// --- No !hip.context arg 0: function left alone (alloc stays). ---
// CHECK-LABEL: func.func @no_context
// CHECK-NOT:     hip.alloc_output
// CHECK:         memref.alloc
func.func @no_context(%x: index) -> memref<?xf16> {
  %out = memref.alloc(%x) : memref<?xf16>
  return %out : memref<?xf16>
}

// --- Defensive: a returned alloc that also has a dealloc -> dealloc erased
//     (the EP-owned output buffer must never be freed). ---
// CHECK-LABEL: func.func @returned_alloc_with_dealloc
// CHECK-NOT:     memref.dealloc
// CHECK:         hip.alloc_output(%{{.*}}, %{{.*}}) {out_idx = 0 : i64} : memref<?xf16>
func.func @returned_alloc_with_dealloc(%ctx: !hip.context, %M: index) -> memref<?xf16> {
  %out = memref.alloc(%M) : memref<?xf16>
  memref.dealloc %out : memref<?xf16>
  return %out : memref<?xf16>
}

// --- Aliased multi-output (same alloc returned twice): exactly one
//     hip.alloc_output (dedupe / no double-erase), both results alias it. ---
// CHECK-LABEL: func.func @aliased_output
// CHECK-COUNT-1: hip.alloc_output
// CHECK-NOT:     hip.alloc_output
// CHECK:         return %[[OUT:.*]], %[[OUT]]
func.func @aliased_output(%ctx: !hip.context, %M: index) -> (memref<?xf16>, memref<?xf16>) {
  %a = memref.alloc(%M) : memref<?xf16>
  return %a, %a : memref<?xf16>, memref<?xf16>
}
