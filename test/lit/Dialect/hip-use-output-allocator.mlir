// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-use-output-allocator.
//
// Checks two things. Step 1: a returned memref.alloc becomes a hip.alloc_output
// that reuses the alloc's dynamic sizes and gets out_idx = its return position.
// Step 2: every OTHER returned value (a returned input, or the same buffer
// returned at a second index) gets its own hip.alloc_output + memref.copy, so
// the allocator callback fires exactly once per output index. Intermediates,
// private helpers, and functions without a !hip.context arg are left alone.
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

// --- Passthrough output (return a memref block-arg): Step 2 makes a fresh
//     alloc_output for the output index and copies the input into it. Without
//     this the EP allocator callback would never fire for this output and the
//     runtime aborts. ---
// CHECK-LABEL: func.func @passthrough
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[X:.*]]: memref<?xf16>)
// CHECK:         %[[D:.*]] = memref.dim %[[X]]
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]], %[[D]]) {out_idx = 0 : i64} : memref<?xf16>
// CHECK:         memref.copy %[[X]], %[[OUT]]
// CHECK:         return %[[OUT]]
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

// --- Same buffer returned twice: Step 1 rewrites the alloc once (at its
//     first-seen return index); Step 2 then gives the OTHER index its own
//     alloc_output + a copy, so each output index is filled exactly once (two
//     separate EP buffers with the same content). ---
// CHECK-LABEL: func.func @aliased_output
// CHECK-DAG:     hip.alloc_output(%{{.*}}) {out_idx = 1 : i64} : memref<?xf16>
// CHECK-DAG:     hip.alloc_output(%{{.*}}) {out_idx = 0 : i64} : memref<?xf16>
// CHECK:         memref.copy
func.func @aliased_output(%ctx: !hip.context, %M: index) -> (memref<?xf16>, memref<?xf16>) {
  %a = memref.alloc(%M) : memref<?xf16>
  return %a, %a : memref<?xf16>, memref<?xf16>
}

// --- Realistic graph with real hip ops (design doc "Add -> MatMul -> Sigmoid").
//     The two intermediates (%t0 = add output, %t1 = matmul output) are NOT
//     returned, so they stay memref.alloc (a later pass pools them). Only the
//     returned sigmoid output is rewritten to hip.alloc_output, reusing the
//     dynamic row count %M and getting out_idx = 0 (its return position). ---
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK:         %[[M:.*]] = memref.dim
// CHECK:         %[[T0:.*]] = memref.alloc(%[[M]]) : memref<?x64xf16>
// CHECK:         hip.add(%[[CTX]]) ins({{.*}}) outs(%[[T0]] : memref<?x64xf16>)
// CHECK:         %[[T1:.*]] = memref.alloc(%[[M]]) : memref<?x64xf16>
// CHECK:         hip.matmul(%[[CTX]]) ins(%[[T0]]{{.*}}) outs(%[[T1]] : memref<?x64xf16>)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]], %[[M]]) {out_idx = 0 : i64} : memref<?x64xf16>
// CHECK:         hip.sigmoid(%[[CTX]]) ins(%[[T1]] : memref<?x64xf16>) outs(%[[OUT]] : memref<?x64xf16>)
// CHECK:         return %[[OUT]]
func.func @main_graph(%ctx: !hip.context,
                      %input: memref<?x64xf16>,
                      %bias: memref<?x64xf16>,
                      %w: memref<64x64xf16>) -> memref<?x64xf16> {
  %c0 = arith.constant 0 : index
  %M = memref.dim %input, %c0 : memref<?x64xf16>
  %t0 = memref.alloc(%M) : memref<?x64xf16>
  hip.add(%ctx) ins(%input, %bias : memref<?x64xf16>, memref<?x64xf16>)
               outs(%t0 : memref<?x64xf16>)
  %t1 = memref.alloc(%M) : memref<?x64xf16>
  hip.matmul(%ctx) ins(%t0, %w : memref<?x64xf16>, memref<64x64xf16>)
                  outs(%t1 : memref<?x64xf16>)
  %out = memref.alloc(%M) : memref<?x64xf16>
  hip.sigmoid(%ctx) ins(%t1 : memref<?x64xf16>) outs(%out : memref<?x64xf16>)
  return %out : memref<?x64xf16>
}

// --- Private function (e.g. an outlined onnx.Loop body): carries !hip.context
//     arg 0 and returns a memref.alloc, but is NOT a public graph entry, so the
//     pass leaves it untouched (its output is DLL-internal, never an EP output).
//     Proves cross-function isolation: publics above are rewritten, this is not. ---
// CHECK-LABEL: func.func private @loop_body
// CHECK-NOT:     hip.alloc_output
// CHECK:         memref.alloc
func.func private @loop_body(%ctx: !hip.context, %M: index) -> memref<?xf16> {
  %out = memref.alloc(%M) : memref<?xf16>
  return %out : memref<?xf16>
}

// --- Public-only guard fires BEFORE dealloc erasure: a PRIVATE function keeps
//     its memref.dealloc (contrast @returned_alloc_with_dealloc, public, where
//     the dealloc IS erased). ---
// CHECK-LABEL: func.func private @priv_with_dealloc
// CHECK:         memref.alloc
// CHECK:         memref.dealloc
// CHECK-NOT:     hip.alloc_output
func.func private @priv_with_dealloc(%ctx: !hip.context, %M: index) -> memref<?xf16> {
  %out = memref.alloc(%M) : memref<?xf16>
  memref.dealloc %out : memref<?xf16>
  return %out : memref<?xf16>
}

// --- Output returned through a shape-adjusting memref.cast: the alloc still
//     becomes hip.alloc_output (reusing its own %M dynamic size + static dims),
//     the cast is left in place feeding the return, and out_idx is the cast's
//     return position. Models the matmul-output buffer whose middle dim is
//     statically known in-graph (memref<?x256x2560>) but declared dynamic in
//     the ONNX output type (memref<?x?x2560>). ---
// CHECK-LABEL: func.func @cast_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[M:.*]]: index)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]], %[[M]]) {out_idx = 0 : i64} : memref<?x256x2560xf16>
// CHECK:         %[[C:.*]] = memref.cast %[[OUT]] : memref<?x256x2560xf16> to memref<?x?x2560xf16>
// CHECK:         return %[[C]]
func.func @cast_output(%ctx: !hip.context, %M: index) -> memref<?x?x2560xf16> {
  %out = memref.alloc(%M) : memref<?x256x2560xf16>
  %c = memref.cast %out : memref<?x256x2560xf16> to memref<?x?x2560xf16>
  return %c : memref<?x?x2560xf16>
}

// --- out_idx follows func.return POSITION, not definition order. %a is defined
//     first but returned at index 1; %b is defined second but returned at index
//     0. Each hip.alloc_output is emitted at its alloc's original site, so the
//     dynamic one (%a, out_idx 1) prints before the static one (%b, out_idx 0). ---
// CHECK-LABEL: func.func @return_order
// CHECK:         hip.alloc_output(%[[CTX:.*]], %{{.*}}) {out_idx = 1 : i64} : memref<?xf16>
// CHECK:         hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<4xf16>
func.func @return_order(%ctx: !hip.context, %M: index) -> (memref<4xf16>, memref<?xf16>) {
  %a = memref.alloc(%M) : memref<?xf16>
  %b = memref.alloc() : memref<4xf16>
  return %b, %a : memref<4xf16>, memref<?xf16>
}
