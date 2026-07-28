// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-use-output-allocator.
//
// Verifies that the pass rewrites graph-output memref.alloc ops (values
// returned by func.return) into hip.alloc_output, reusing the alloc's dynamic
// sizes and setting out_idx to the return position, while leaving intermediates,
// passthrough outputs, private helpers, and context-less functions untouched.
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

// --- Output returned through memref.collapse_shape: the alloc still becomes
//     hip.alloc_output, and the collapse_shape stays in place feeding the
//     return. Models a reshaped output such as a rank-4 conv result flattened
//     to rank-2 before the return, so the buffer is owned by the EP rather than
//     pooled. Because the return type (rank 2) is lower-rank than the buffer
//     (rank 4), the pass stamps hipdnn.abi_shape / hipdnn.abi_groups so the
//     later lowering can hand the runtime the external (ONNX ABI) shape instead
//     of the internal rank-4 descriptor. Here abi_groups=[1,3] means external
//     dim0 = internal dim0 and external dim1 = product of internal dims 1..3. ---
// CHECK-LABEL: func.func @collapse_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {hipdnn.abi_groups = array<i64: 1, 3>, hipdnn.abi_shape = array<i64: 1, 200704>, out_idx = 0 : i64} : memref<1x64x56x56xf32>
// CHECK:         %[[C:.*]] = memref.collapse_shape %[[OUT]]
// CHECK:         return %[[C]]
func.func @collapse_output(%ctx: !hip.context) -> memref<1x200704xf32> {
  %x = memref.alloc() : memref<1x64x56x56xf32>
  %out = memref.collapse_shape %x [[0], [1, 2, 3]]
       : memref<1x64x56x56xf32> into memref<1x200704xf32>
  return %out : memref<1x200704xf32>
}

// --- Output returned through memref.expand_shape: rank-increasing view; the
//     pass stamps abi_groups per INTERNAL dim (expand reassociation). ---
// CHECK-LABEL: func.func @expand_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {hipdnn.abi_groups = array<i64: 1, 3>, hipdnn.abi_shape = array<i64: 1, 64, 56, 56>, out_idx = 0 : i64} : memref<1x200704xf32>
// CHECK:         %[[E:.*]] = memref.expand_shape %[[OUT]]
// CHECK:         return %[[E]]
func.func @expand_output(%ctx: !hip.context) -> memref<1x64x56x56xf32> {
  %x = memref.alloc() : memref<1x200704xf32>
  %out = memref.expand_shape %x [[0], [1, 2, 3]]
       output_shape [1, 64, 56, 56]
       : memref<1x200704xf32> into memref<1x64x56x56xf32>
  return %out : memref<1x64x56x56xf32>
}

// --- A chain of view ops (collapse_shape -> cast) before the return: the alloc
//     is still converted, exactly once. Shows that multi-op view chains are
//     followed, not just a single view op. The rank-reducing collapse (rank 2 ->
//     rank 1) is stamped for the lowering: abi_groups=[2] means external dim0 =
//     product of internal dims 0..1 (2*4=8); abi_shape is dynamic because the
//     returned type is memref<?xf32> (the cast erased the static extent). ---
// CHECK-LABEL: func.func @chain_output
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%{{.*}}) {hipdnn.abi_groups = array<i64: 2>, hipdnn.abi_shape = array<i64: {{-?[0-9]+}}>, out_idx = 0 : i64} : memref<2x4xf32>
// CHECK:         %[[COL:.*]] = memref.collapse_shape %[[OUT]]
// CHECK:         %[[CST:.*]] = memref.cast %[[COL]]
// CHECK:         return %[[CST]]
func.func @chain_output(%ctx: !hip.context) -> memref<?xf32> {
  %x = memref.alloc() : memref<2x4xf32>
  %col = memref.collapse_shape %x [[0, 1]]
       : memref<2x4xf32> into memref<8xf32>
  %cast = memref.cast %col : memref<8xf32> to memref<?xf32>
  return %cast : memref<?xf32>
}

// --- A longer 3-op view chain (collapse_shape -> expand_shape -> cast), mixing
//     all three view-op kinds before the return: the root alloc is still
//     converted exactly once, and the whole chain is left in place. Proves the
//     alias analysis follows arbitrary-length chains, not just one or two ops. ---
// CHECK-LABEL: func.func @long_chain_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<2x4x8xf32>
// CHECK:         %[[COL:.*]] = memref.collapse_shape %[[OUT]]
// CHECK:         %[[EXP:.*]] = memref.expand_shape %[[COL]]
// CHECK:         %[[CST:.*]] = memref.cast %[[EXP]]
// CHECK:         return %[[CST]]
func.func @long_chain_output(%ctx: !hip.context) -> memref<?x8xf32> {
  %x = memref.alloc() : memref<2x4x8xf32>
  %col = memref.collapse_shape %x [[0, 1, 2]]
       : memref<2x4x8xf32> into memref<64xf32>
  %exp = memref.expand_shape %col [[0, 1]] output_shape [8, 8]
       : memref<64xf32> into memref<8x8xf32>
  %cast = memref.cast %exp : memref<8x8xf32> to memref<?x8xf32>
  return %cast : memref<?x8xf32>
}

// --- An even longer 5-op view chain exercising every view-op kind the analysis
//     handles (collapse_shape -> collapse_shape -> expand_shape -> subview ->
//     cast): the root alloc is still converted exactly once and the full chain
//     is preserved. Stresses that the backward alias walk terminates correctly
//     no matter how deep the view chain is. ---
// CHECK-LABEL: func.func @longer_chain_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<2x4x8xf32>
// CHECK:         %[[C1:.*]] = memref.collapse_shape %[[OUT]]
// CHECK:         %[[C2:.*]] = memref.collapse_shape %[[C1]]
// CHECK:         %[[E:.*]] = memref.expand_shape %[[C2]]
// CHECK:         %[[S:.*]] = memref.subview %[[E]]
// CHECK:         %[[CST:.*]] = memref.cast %[[S]]
// CHECK:         return %[[CST]]
func.func @longer_chain_output(%ctx: !hip.context) -> memref<?x8xf32, strided<[8, 1]>> {
  %x = memref.alloc() : memref<2x4x8xf32>
  %c1 = memref.collapse_shape %x [[0, 1], [2]]
      : memref<2x4x8xf32> into memref<8x8xf32>
  %c2 = memref.collapse_shape %c1 [[0, 1]]
      : memref<8x8xf32> into memref<64xf32>
  %e = memref.expand_shape %c2 [[0, 1]] output_shape [8, 8]
     : memref<64xf32> into memref<8x8xf32>
  %s = memref.subview %e[0, 0] [4, 8] [1, 1]
     : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1]>>
  %cast = memref.cast %s
        : memref<4x8xf32, strided<[8, 1]>> to memref<?x8xf32, strided<[8, 1]>>
  return %cast : memref<?x8xf32, strided<[8, 1]>>
}

// --- Output returned through memref.subview: the parent alloc becomes
//     hip.alloc_output; the subview is left in place feeding the return. ---
// CHECK-LABEL: func.func @subview_output
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64} : memref<4x8xf32>
// CHECK:         %[[S:.*]] = memref.subview %[[OUT]]
// CHECK:         return %[[S]]
func.func @subview_output(%ctx: !hip.context) -> memref<2x4xf32, strided<[8, 1]>> {
  %x = memref.alloc() : memref<4x8xf32>
  %s = memref.subview %x[0, 0] [2, 4] [1, 1]
     : memref<4x8xf32> to memref<2x4xf32, strided<[8, 1]>>
  return %s : memref<2x4xf32, strided<[8, 1]>>
}

// --- DETR-class expand: internal rank-2 Gemm buffer, graph output rank 3.
//     abi_groups=[2,1]: internal dim0 -> external dims [0,1]; internal dim1 ->
//     external dim [2]. Matches PR #557 logits scenario in abi_shape style.
// CHECK-LABEL: func.func @expand_logits_class
// CHECK:         hip.alloc_output(%{{.*}}) {hipdnn.abi_groups = array<i64: 2, 1>, hipdnn.abi_shape = array<i64: 1, {{-?[0-9]+}}, 92>, out_idx = 0 : i64} : memref<?x92xf16>
func.func @expand_logits_class(%ctx: !hip.context, %n: index) -> memref<1x?x92xf16> {
  %out = memref.alloc(%n) : memref<?x92xf16>
  %ret = memref.expand_shape %out [[0, 1], [2]] output_shape [1, %n, 92]
       : memref<?x92xf16> into memref<1x?x92xf16>
  return %ret : memref<1x?x92xf16>
}
