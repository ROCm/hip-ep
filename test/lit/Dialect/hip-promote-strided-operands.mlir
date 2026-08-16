// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-promote-strided-operands.
//
// The pass inserts memref.alloc + memref.copy + memref.dealloc for each
// non-identity-layout DPS-input memref, regardless of the operation's dialect.
// DPS-init operands and identity-layout inputs are left untouched.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-promote-strided-operands %s | FileCheck %s

// ----------------------------------------------------------------------------
// Positive: a non-identity-layout subview feeding hip.sigmoid is copied into
// an identity-layout allocation, consumed, then deallocated.
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
// Negative: a hip.sigmoid with an identity-layout input is not rewritten.
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
// Multiple non-identity-layout inputs: a hip.add with two subview operands gets
// two independent promotions (separate allocs, copies, and deallocs).
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
// DPS-init operands are outside this pass's scope. Even when presented with a
// non-identity layout, the init is not rewritten: safe output promotion would
// require copy-back and preservation of any read-before-write semantics.
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
// Models the pattern from hip-optimize-memrefs buffer-reuse: a single backing
// allocation can be re-bound at multiple offsets for non-overlapping live
// ranges. When the resulting reinterpret_cast carries a non-zero offset, the
// pass must materialize an identity-layout temporary just like the subview
// case.
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
// Memory-space preservation: the promoted allocation must inherit the source
// memory space rather than silently falling back to the default.
//
// makeIdentityLayoutType preserves sourceType.getMemorySpace(); this test pins
// that behavior across future refactors.
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

// ----------------------------------------------------------------------------
// MatMul lowering extracts bare pointers, so both DPS-input matrices must have
// identity layout. This case exercises non-contiguous strides for A and
// canonical strides with a non-zero offset for B. The identity-layout DPS init
// remains unchanged.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @matmul_promotes_strided_inputs
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context,
// CHECK-SAME:     %[[A_PARENT:.*]]: memref<2x8xf16>,
// CHECK-SAME:     %[[B_PARENT:.*]]: memref<8x3xf16>,
// CHECK-SAME:     %[[OUT:.*]]: memref<2x3xf16>)
// CHECK:         %[[A_SV:.*]] = memref.subview %[[A_PARENT]]
// CHECK:         %[[B_SV:.*]] = memref.subview %[[B_PARENT]]
// CHECK:         %[[A_TMP:.*]] = memref.alloc() : memref<2x4xf16>
// CHECK:         memref.copy %[[A_SV]], %[[A_TMP]]
// CHECK:         %[[B_TMP:.*]] = memref.alloc() : memref<4x3xf16>
// CHECK:         memref.copy %[[B_SV]], %[[B_TMP]]
// CHECK:         hip.matmul(%[[CTX]])
// CHECK-SAME:      ins(%[[A_TMP]], %[[B_TMP]] : memref<2x4xf16>, memref<4x3xf16>)
// CHECK-SAME:      outs(%[[OUT]] : memref<2x3xf16>)
// CHECK:         memref.dealloc %[[A_TMP]]
// CHECK:         memref.dealloc %[[B_TMP]]
// CHECK:         return
func.func @matmul_promotes_strided_inputs(
    %ctx: !hip.context,
    %a_parent: memref<2x8xf16>,
    %b_parent: memref<8x3xf16>,
    %out: memref<2x3xf16>) {
  %a = memref.subview %a_parent[0, 4][2, 4][1, 1]
      : memref<2x8xf16>
        to memref<2x4xf16, strided<[8, 1], offset: 4>>
  %b = memref.subview %b_parent[4, 0][4, 3][1, 1]
      : memref<8x3xf16>
        to memref<4x3xf16, strided<[3, 1], offset: 12>>
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, strided<[8, 1], offset: 4>>,
                 memref<4x3xf16, strided<[3, 1], offset: 12>>)
    outs(%out : memref<2x3xf16>)
  return
}

// ----------------------------------------------------------------------------
// Interface selection: linalg.generic is the lightest non-HIP DPS operation
// available in this test. The production pipeline lowers linalg before this
// pass, but the isolated pass deliberately selects by
// DestinationStyleOpInterface rather than by dialect or operation name.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @non_hip_dps_input_promoted
// CHECK-SAME:    (%[[SRC:.*]]: memref<2x4xf32>,
// CHECK-SAME:     %[[OUT:.*]]: memref<2x2xf32>)
// CHECK:         %[[SV:.*]] = memref.subview %[[SRC]]
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK:         memref.copy %[[SV]], %[[TMP]]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[TMP]] : memref<2x2xf32>)
// CHECK-SAME:      outs(%[[OUT]] : memref<2x2xf32>)
// CHECK:         memref.dealloc %[[TMP]]
// CHECK:         return
func.func @non_hip_dps_input_promoted(
    %src: memref<2x4xf32>,
    %out: memref<2x2xf32>) {
  %sv = memref.subview %src[0, 2][2, 2][1, 1]
      : memref<2x4xf32>
        to memref<2x2xf32, strided<[4, 1], offset: 2>>
  linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%sv : memref<2x2xf32, strided<[4, 1], offset: 2>>)
    outs(%out : memref<2x2xf32>) {
  ^bb0(%in: f32, %unused: f32):
    linalg.yield %in : f32
  }
  return
}

// ----------------------------------------------------------------------------
// Loop captures use the outlined body signature as their ABI contract. Repeated
// read-only captures of the same strided value share one identity-layout copy,
// which remains live through the invocation and is deallocated immediately
// afterward. The carrier seed is not promoted or deallocated: a zero-trip loop
// may return that borrowed descriptor.
//
// The input intentionally models the verifier-supported transient
// post-bufferization layout mismatch repaired by this pass.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @loop_capture_promoted_once
// CHECK-SAME:    %[[SEED:[^ ,]+]]: memref<4x4xf32>
// CHECK:         %[[CAPTURE:.*]] = memref.subview
// CHECK:         %[[TMP:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK-NEXT:    memref.copy %[[CAPTURE]], %[[TMP]]
// CHECK-NEXT:    %[[LOOP:.*]]:2 = hip.loop
// CHECK-SAME:      iter_args(%[[SEED]] : memref<4x4xf32>)
// CHECK-SAME:      captures(%[[TMP]], %[[TMP]] : memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NEXT:    memref.dealloc %[[TMP]]
// CHECK-NEXT:    memref.copy %[[LOOP]]#0
// CHECK-NOT:     memref.dealloc %[[SEED]]
func.func private @loop_capture_promoted_once_body(
    %ctx: !hip.context, %iter: memref<i64>, %cond: memref<i1>,
    %current: memref<4x4xf32>, %capture0: memref<4x4xf32>,
    %capture1: memref<4x4xf32>, %frame: !hip.loop_frame)
    -> (i32, memref<4x4xf32>) {
  %status = arith.constant 0 : i32
  return %status, %current : i32, memref<4x4xf32>
}

func.func @loop_capture_promoted_once(
    %ctx: !hip.context, %parent: memref<4x8xf32>,
    %seed: memref<4x4xf32>, %after: memref<4x4xf32>) {
  %zero = arith.constant 0 : index
  %true = arith.constant true
  %capture = memref.subview %parent[0, 2][4, 4][1, 1]
      : memref<4x8xf32>
        to memref<4x4xf32, strided<[8, 1], offset: 2>>
  %result, %loop_frame = hip.loop(%ctx, %zero, %true)
      iter_args(%seed : memref<4x4xf32>)
      captures(%capture, %capture
        : memref<4x4xf32, strided<[8, 1], offset: 2>>,
          memref<4x4xf32, strided<[8, 1], offset: 2>>)
      -> (memref<4x4xf32>, !hip.loop_frame)
      body @loop_capture_promoted_once_body
      {cond_is_passthrough, descriptor_return,
       num_loop_carried = 1 : i32}
  memref.copy %result, %after : memref<4x4xf32> to memref<4x4xf32>
  hip.loop_frame_destroy(%ctx, %loop_frame)
  return
}
