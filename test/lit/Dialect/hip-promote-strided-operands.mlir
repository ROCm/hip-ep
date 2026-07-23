// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-promote-strided-operands.
//
// The pass inserts memref.alloc + memref.copy + memref.dealloc for any
// DPS-input memref operand of a hip.* op whose layout is non-identity
// (non-zero offset or non-contiguous strides).  DPS-init (output) operands
// and already-contiguous inputs are left untouched.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-promote-strided-operands %s | FileCheck %s

// ----------------------------------------------------------------------------
// Positive: strided subview feeding hip.sigmoid is promoted to a contiguous
// alloc, copied in, consumed, then deallocated after the consumer.
//
// Models the bug pattern from onnx.Split -> onnx.Sigmoid: bufferization
// produces memref.subview with offset/stride layout, which would otherwise
// be silently flattened to a base pointer at LLVM lowering time.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @sigmoid_promotes_strided_input
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:     %[[SRC:.*]]: memref<1x128x2048xf16>,
// CHECK-SAME:     %[[OUT:.*]]: memref<1x128x1024xf16>)
// CHECK:         %[[SV:.*]] = memref.subview %[[SRC]]
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[SV]], %[[TMP]]
// CHECK:         hip.sigmoid(%[[CTX]]) ins(%[[TMP]] : memref<1x128x1024xf16>) outs(%[[OUT]] : memref<1x128x1024xf16>)
// CHECK:         memref.dealloc %[[TMP]]
// CHECK:         return
func.func @sigmoid_promotes_strided_input(
    %ctx: !hip.context,
    %src: memref<1x128x2048xf16>,
    %out: memref<1x128x1024xf16>) {
  %sv = memref.subview %src[0, 0, 1024][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>
  hip.sigmoid(%ctx)
    ins(%sv : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>)
    outs(%out : memref<1x128x1024xf16>)
  return
}

// ----------------------------------------------------------------------------
// Negative: a hip.sigmoid with an already-contiguous (identity-layout) input
// is not rewritten.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @sigmoid_contiguous_input_untouched
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:     %[[IN:.*]]: memref<1x128x1024xf16>,
// CHECK-SAME:     %[[OUT:.*]]: memref<1x128x1024xf16>)
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         hip.sigmoid(%[[CTX]]) ins(%[[IN]] : memref<1x128x1024xf16>) outs(%[[OUT]] : memref<1x128x1024xf16>)
// CHECK-NOT:     memref.dealloc
// CHECK:         return
func.func @sigmoid_contiguous_input_untouched(
    %ctx: !hip.context,
    %in: memref<1x128x1024xf16>,
    %out: memref<1x128x1024xf16>) {
  hip.sigmoid(%ctx) ins(%in : memref<1x128x1024xf16>)
                    outs(%out : memref<1x128x1024xf16>)
  return
}

// ----------------------------------------------------------------------------
// Multiple strided inputs: a hip.add with two subview operands gets two
// independent promotions (separate allocs, copies, and deallocs).
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @add_promotes_both_inputs
// CHECK:         %[[SVA:.*]] = memref.subview
// CHECK:         %[[SVB:.*]] = memref.subview
// CHECK:         %[[TA:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[SVA]], %[[TA]]
// CHECK:         %[[TB:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[SVB]], %[[TB]]
// CHECK:         hip.add{{.*}}ins(%[[TA]], %[[TB]]
// CHECK:         memref.dealloc %[[TA]]
// CHECK:         memref.dealloc %[[TB]]
func.func @add_promotes_both_inputs(
    %ctx: !hip.context,
    %a: memref<1x128x2048xf16>,
    %b: memref<1x128x2048xf16>,
    %out: memref<1x128x1024xf16>) {
  %sva = memref.subview %a[0, 0, 0][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1]>>
  %svb = memref.subview %b[0, 0, 1024][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>
  hip.add(%ctx)
    ins(%sva, %svb : memref<1x128x1024xf16, strided<[262144, 2048, 1]>>,
                     memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>)
    outs(%out : memref<1x128x1024xf16>)
  return
}

// ----------------------------------------------------------------------------
// DPS-init (output) operand is left untouched even when its type carries an
// explicit strided layout.  Outs are guaranteed contiguous by construction in
// this pipeline; promoting them would also change observable behavior because
// the consumer writes through the operand (the strided slice is the intended
// destination, not a temporary to copy back).
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @strided_output_left_alone
// CHECK:         %[[SV:.*]] = memref.subview
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         hip.sigmoid({{.*}}) ins({{.*}}) outs(%[[SV]]
// CHECK-NOT:     memref.dealloc
func.func @strided_output_left_alone(
    %ctx: !hip.context,
    %in: memref<1x128x1024xf16>,
    %dst: memref<1x128x2048xf16>) {
  %sv_out = memref.subview %dst[0, 0, 1024][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>
  hip.sigmoid(%ctx)
    ins(%in : memref<1x128x1024xf16>)
    outs(%sv_out : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>)
  return
}

// ----------------------------------------------------------------------------
// Function-argument with a strided layout (no defining op) is still promoted.
//
// This is the key generality difference vs upstream linalg::promoteSubViews,
// which keys on `isa<memref::SubViewOp>(operand.getDefiningOp())` and would
// miss this case.  Our predicate is type-based, so any non-identity layout
// triggers promotion regardless of how it was produced.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @func_arg_strided_promoted
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:     %[[SRC:.*]]: memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>,
// CHECK-SAME:     %[[OUT:.*]]: memref<1x128x1024xf16>)
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[SRC]], %[[TMP]]
// CHECK:         hip.sigmoid(%[[CTX]]) ins(%[[TMP]] : memref<1x128x1024xf16>) outs(%[[OUT]] : memref<1x128x1024xf16>)
// CHECK:         memref.dealloc %[[TMP]]
func.func @func_arg_strided_promoted(
    %ctx: !hip.context,
    %src: memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>,
    %out: memref<1x128x1024xf16>) {
  hip.sigmoid(%ctx)
    ins(%src : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>)
    outs(%out : memref<1x128x1024xf16>)
  return
}

// ----------------------------------------------------------------------------
// memref.reinterpret_cast with a non-zero offset produces a non-identity
// layout that must be promoted.
//
// Models a buffer-reuse pattern: a single backing
// allocation can be re-bound at multiple offsets for non-overlapping live
// ranges.  When the resulting reinterpret_cast carries a non-zero offset, our
// pass must materialize a contiguous temporary just like the subview case.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @reinterpret_cast_with_offset_promoted
// CHECK:         %[[RC:.*]] = memref.reinterpret_cast
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[RC]], %[[TMP]]
// CHECK:         hip.sigmoid({{.*}}) ins(%[[TMP]] : memref<1x128x1024xf16>)
// CHECK:         memref.dealloc %[[TMP]]
func.func @reinterpret_cast_with_offset_promoted(
    %ctx: !hip.context,
    %src: memref<1x128x4096xf16>,
    %out: memref<1x128x1024xf16>) {
  %rc = memref.reinterpret_cast %src
      to offset: [131072], sizes: [1, 128, 1024], strides: [131072, 1024, 1]
      : memref<1x128x4096xf16>
        to memref<1x128x1024xf16, strided<[131072, 1024, 1], offset: 131072>>
  hip.sigmoid(%ctx)
    ins(%rc : memref<1x128x1024xf16, strided<[131072, 1024, 1], offset: 131072>>)
    outs(%out : memref<1x128x1024xf16>)
  return
}

// ----------------------------------------------------------------------------
// Dynamic shape: collectDynamicSizes() must emit one memref.dim per dynamic
// dimension and forward those sizes to memref.alloc.
//
// Source has a dynamic outer dim and a dynamic offset (after subview).  The
// promoted alloc keeps the dynamic shape under identity layout and needs the
// dim sizes as operands.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @dynamic_shape_promoted
// CHECK:         %[[SV:.*]] = memref.subview
// CHECK:         %[[DIM:.*]] = memref.dim %[[SV]]
// CHECK:         %[[TMP:.*]] = memref.alloc(%[[DIM]]) : memref<?x128xf16>
// CHECK:         memref.copy %[[SV]], %[[TMP]]
// CHECK:         hip.sigmoid({{.*}}) ins(%[[TMP]] : memref<?x128xf16>)
// CHECK:         memref.dealloc %[[TMP]]
func.func @dynamic_shape_promoted(
    %ctx: !hip.context,
    %src: memref<?x128xf16>,
    %out: memref<?x128xf16>,
    %off: index,
    %sz: index) {
  %sv = memref.subview %src[%off, 0][%sz, 128][1, 1]
      : memref<?x128xf16>
        to memref<?x128xf16, strided<[128, 1], offset: ?>>
  hip.sigmoid(%ctx)
    ins(%sv : memref<?x128xf16, strided<[128, 1], offset: ?>>)
    outs(%out : memref<?x128xf16>)
  return
}

// ----------------------------------------------------------------------------
// Memory-space preservation: the promoted alloc must inherit the memory
// space of the strided source, not silently fall back to the default space.
//
// makeContiguousType(src) preserves src.getMemorySpace() — this test pins the
// behavior so a future refactor of that helper can't drop memory spaces.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @memory_space_preserved
// CHECK:         memref.alloc() : memref<1x128x1024xf16, 1>
func.func @memory_space_preserved(
    %ctx: !hip.context,
    %src: memref<1x128x2048xf16, 1>,
    %out: memref<1x128x1024xf16, 1>) {
  %sv = memref.subview %src[0, 0, 1024][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16, 1>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>, 1>
  hip.sigmoid(%ctx)
    ins(%sv : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>, 1>)
    outs(%out : memref<1x128x1024xf16, 1>)
  return
}

// ----------------------------------------------------------------------------
// Mixed inputs: only the strided operand is promoted; the already-contiguous
// operand is passed through unchanged.  Validates that the per-operand
// decision in the input loop doesn't over- or under-promote.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @mixed_strided_and_contiguous_inputs
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:     %[[A:.*]]: memref<1x128x2048xf16>,
// CHECK-SAME:     %[[B:.*]]: memref<1x128x1024xf16>,
// CHECK-SAME:     %[[OUT:.*]]: memref<1x128x1024xf16>)
// CHECK:         %[[SV:.*]] = memref.subview %[[A]]
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<1x128x1024xf16>
// CHECK:         memref.copy %[[SV]], %[[TMP]]
// CHECK-NOT:     memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         hip.add(%[[CTX]]) ins(%[[TMP]], %[[B]]
// CHECK:         memref.dealloc %[[TMP]]
// CHECK-NOT:     memref.dealloc
func.func @mixed_strided_and_contiguous_inputs(
    %ctx: !hip.context,
    %a: memref<1x128x2048xf16>,
    %b: memref<1x128x1024xf16>,
    %out: memref<1x128x1024xf16>) {
  %sv = memref.subview %a[0, 0, 1024][1, 128, 1024][1, 1, 1]
      : memref<1x128x2048xf16>
        to memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>
  hip.add(%ctx)
    ins(%sv, %b : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>,
                  memref<1x128x1024xf16>)
    outs(%out : memref<1x128x1024xf16>)
  return
}
