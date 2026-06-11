// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-relax-multi-dyn-expand-shape.
//
// Verifies that the pass rewrites memref.expand_shape ops whose reassociation
// has any group with MORE THAN ONE dynamic output dim into
// memref.reinterpret_cast with explicit sizes/strides — and leaves every
// other shape op (single-dyn-per-group expand_shape, all-static expand_shape,
// and every collapse_shape) untouched.
//
// Workaround for upstream `expand-strided-metadata` asserting on multi-dyn
// groups; see RelaxMultiDynExpandShape.cpp header for context.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-relax-multi-dyn-expand-shape %s 2>&1 | FileCheck %s

// --- Canonical case: ONNX `Range -> Reshape([bs, ss])` materialising 2-D
//     position_ids.  Source memref<?xi64> is identity-layout (stride 1).
//     Single reassociation group [[0, 1]] with BOTH dyn → must rewrite. ---
// CHECK-LABEL: func.func @range_reshape_2d
// CHECK-SAME:    (%[[SRC:.*]]: memref<?xi64>, %[[BS:.*]]: index, %[[SS:.*]]: index)
// CHECK-NOT:     memref.expand_shape
// CHECK:         memref.reinterpret_cast %[[SRC]]
// CHECK-SAME:      to offset: [0], sizes: [%[[BS]], %[[SS]]], strides: [%[[SS]], 1]
// CHECK-SAME:      memref<?xi64> to memref<?x?xi64>
func.func @range_reshape_2d(%src: memref<?xi64>, %bs: index, %ss: index)
    -> memref<?x?xi64> {
  %expand = memref.expand_shape %src [[0, 1]]
            output_shape [%bs, %ss]
            : memref<?xi64> into memref<?x?xi64>
  return %expand : memref<?x?xi64>
}

// --- 1-D -> 3-D, all three output dims dynamic in one group.  Stride chain:
//       stride[2] = 1 (src stride[0])
//       stride[1] = stride[2] * size[2] = %z
//       stride[0] = stride[1] * size[1] = %z * %y  → an affine.apply (the
//       running stride * size product; `makeComposedFoldedAffineApply` emits
//       `affine_map<()[s0, s1] -> (s0 * s1)>` since both operands are dynamic)
// CHECK-LABEL: func.func @one_to_three_all_dyn
// CHECK-SAME:    (%[[SRC:.*]]: memref<?xi64>, %[[X:.*]]: index, %[[Y:.*]]: index, %[[Z:.*]]: index)
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[YZ:.*]] = affine.apply {{.*}}()[%[[Z]], %[[Y]]]
// CHECK:         memref.reinterpret_cast %[[SRC]]
// CHECK-SAME:      to offset: [0], sizes: [%[[X]], %[[Y]], %[[Z]]], strides: [%[[YZ]], %[[Z]], 1]
// CHECK-SAME:      memref<?xi64> to memref<?x?x?xi64>
func.func @one_to_three_all_dyn(%src: memref<?xi64>,
                                %x: index, %y: index, %z: index)
    -> memref<?x?x?xi64> {
  %expand = memref.expand_shape %src [[0, 1, 2]]
            output_shape [%x, %y, %z]
            : memref<?xi64> into memref<?x?x?xi64>
  return %expand : memref<?x?x?xi64>
}

// --- Mixed dyn + static in a single multi-dyn group.  Source 1-D, result
//     3-D with output_shape = [%a, 4, %b] → still two dyn outputs in one
//     group, so it matches.  Stride chain:
//       stride[2] = 1
//       stride[1] = stride[2] * size[2] = %b
//       stride[0] = stride[1] * size[1] = %b * 4 → an affine.apply with the
//       static factor folded into the map (`affine_map<()[s0] -> (s0 * 4)>`)
// CHECK-LABEL: func.func @mixed_dyn_static
// CHECK-SAME:    (%[[SRC:.*]]: memref<?xi64>, %[[A:.*]]: index, %[[B:.*]]: index)
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[B4:.*]] = affine.apply {{.*}}()[%[[B]]]
// CHECK:         memref.reinterpret_cast %[[SRC]]
// CHECK-SAME:      to offset: [0], sizes: [%[[A]], 4, %[[B]]], strides: [%[[B4]], %[[B]], 1]
// CHECK-SAME:      memref<?xi64> to memref<?x4x?xi64>
func.func @mixed_dyn_static(%src: memref<?xi64>, %a: index, %b: index)
    -> memref<?x4x?xi64> {
  %expand = memref.expand_shape %src [[0, 1, 2]]
            output_shape [%a, 4, %b]
            : memref<?xi64> into memref<?x4x?xi64>
  return %expand : memref<?x4x?xi64>
}

// --- Multi-rank source: 2-D source, expand to 4-D with one multi-dyn group
//     [[0, 1], [2, 3]].  Group 0 has 2 dyn dims → MATCH.  Group 1 is
//     all-static — we still emit strides for both groups in the rewritten
//     reinterpret_cast.  Source memref<?x4xi64> identity layout has
//     strides [4, 1].
//       Group 0: inner=1, mixedStrides[1] = 4 (src stride[0])
//                i=0:  mixedStrides[0] = 4 * size[1] = 4 * %ss → affine.apply
//       Group 1: inner=3, mixedStrides[3] = 1 (src stride[1])
//                i=2:  mixedStrides[2] = 1 * size[3] = 2 (fully folded)
// CHECK-LABEL: func.func @two_to_four_one_multi_dyn_group
// CHECK-SAME:    (%[[SRC:.*]]: memref<?x4xi64>, %[[BS:.*]]: index, %[[SS:.*]]: index)
// CHECK-NOT:     memref.expand_shape
// CHECK:         %[[SS4:.*]] = affine.apply {{.*}}()[%[[SS]]]
// CHECK:         memref.reinterpret_cast %[[SRC]]
// CHECK-SAME:      to offset: [0], sizes: [%[[BS]], %[[SS]], 2, 2], strides: [%[[SS4]], 4, 2, 1]
// CHECK-SAME:      memref<?x4xi64> to memref<?x?x2x2xi64>
func.func @two_to_four_one_multi_dyn_group(%src: memref<?x4xi64>,
                                            %bs: index, %ss: index)
    -> memref<?x?x2x2xi64> {
  %expand = memref.expand_shape %src [[0, 1], [2, 3]]
            output_shape [%bs, %ss, 2, 2]
            : memref<?x4xi64> into memref<?x?x2x2xi64>
  return %expand : memref<?x?x2x2xi64>
}

// --- Static non-zero offset is preserved.  Source carries `offset: 8`; the
//     rewritten reinterpret_cast must thread that same static offset through
//     (the pass only bails when the offset is *dynamic*).  Single group [[0,1]]
//     with both dyn → match. ---
// CHECK-LABEL: func.func @nonzero_static_offset
// CHECK-SAME:    (%[[SRC:.*]]: memref<?xi64, strided<[1], offset: 8>>, %[[BS:.*]]: index, %[[SS:.*]]: index)
// CHECK-NOT:     memref.expand_shape
// CHECK:         memref.reinterpret_cast %[[SRC]]
// CHECK-SAME:      to offset: [8], sizes: [%[[BS]], %[[SS]]], strides: [%[[SS]], 1]
func.func @nonzero_static_offset(%src: memref<?xi64, strided<[1], offset: 8>>,
                                 %bs: index, %ss: index)
    -> memref<?x?xi64, strided<[?, 1], offset: 8>> {
  %expand = memref.expand_shape %src [[0, 1]]
            output_shape [%bs, %ss]
            : memref<?xi64, strided<[1], offset: 8>>
              into memref<?x?xi64, strided<[?, 1], offset: 8>>
  return %expand : memref<?x?xi64, strided<[?, 1], offset: 8>>
}

// --- Negative: single-dyn-per-group expand_shape.  Upstream
//     `expand-strided-metadata` handles this correctly; the pass must NOT
//     rewrite.  Source memref<?xi64> [[0, 1]] output_shape [%bs, 128]:
//     group has exactly one dyn out (the result's `?`), so dynCount = 1
//     → skip.
// CHECK-LABEL: func.func @single_dyn_per_group_left_alone
// CHECK:         memref.expand_shape
// CHECK-NOT:     memref.reinterpret_cast
func.func @single_dyn_per_group_left_alone(%src: memref<?xi64>, %bs: index)
    -> memref<?x128xi64> {
  %expand = memref.expand_shape %src [[0, 1]]
            output_shape [%bs, 128]
            : memref<?xi64> into memref<?x128xi64>
  return %expand : memref<?x128xi64>
}

// --- Negative: all-static expand_shape.  No dyn dim anywhere → skip.
// CHECK-LABEL: func.func @all_static_left_alone
// CHECK:         memref.expand_shape
// CHECK-NOT:     memref.reinterpret_cast
func.func @all_static_left_alone(%src: memref<256xi64>)
    -> memref<4x64xi64> {
  %expand = memref.expand_shape %src [[0, 1]]
            output_shape [4, 64]
            : memref<256xi64> into memref<4x64xi64>
  return %expand : memref<4x64xi64>
}

// --- Negative: memref.collapse_shape — never rewritten by this pass even
//     if it has multiple dyn dims.  Upstream handles collapse fine. ---
// CHECK-LABEL: func.func @collapse_shape_left_alone
// CHECK:         memref.collapse_shape
// CHECK-NOT:     memref.reinterpret_cast
func.func @collapse_shape_left_alone(%src: memref<?x?xi64>) -> memref<?xi64> {
  %c = memref.collapse_shape %src [[0, 1]]
       : memref<?x?xi64> into memref<?xi64>
  return %c : memref<?xi64>
}

// --- Negative: dynamic stride source.  Pass conservatively bails when
//     source strides aren't known — leaves the op for upstream to fail loudly
//     rather than silently miscompiling. ---
// CHECK-LABEL: func.func @dynamic_stride_src_bails
// CHECK:         memref.expand_shape
// CHECK-NOT:     memref.reinterpret_cast
func.func @dynamic_stride_src_bails(
    %src: memref<?xi64, strided<[?], offset: 0>>,
    %bs: index, %ss: index) -> memref<?x?xi64, strided<[?, ?], offset: 0>> {
  %expand = memref.expand_shape %src [[0, 1]]
            output_shape [%bs, %ss]
            : memref<?xi64, strided<[?], offset: 0>>
              into memref<?x?xi64, strided<[?, ?], offset: 0>>
  return %expand : memref<?x?xi64, strided<[?, ?], offset: 0>>
}
