// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for the view-passthrough enhancement in
// --hip-annotate-input-dim-slots (Phase 2.6 of slot-buffer-coalesce).
//
// The pass now walks back through view-style memref ops
// (memref.subview, memref.cast, memref.view, memref.reinterpret_cast,
// memref.expand_shape, memref.collapse_shape) to find the real DPS
// writer of an operand. Without this, a chain like
//
//   hip.nonzero(...) outs(%pool_view)
//   %view = memref.expand_shape %pool_view
//   hip.transpose ins(%view)
//
// would fail to discover hip.nonzero as the producer of %view's
// underlying buffer, so the transpose would miss its dim-slot
// annotation and the lowering would read upper-bound dims from the
// descriptor.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-compose-dim-specs --hip-annotate-input-dim-slots %s | FileCheck %s

// ===== NonZero -> memref.expand_shape -> hip.transpose =====
//
// The expand_shape inserts a leading unit dim. After the view walk,
// hip.transpose should still discover hip.nonzero as the writer and
// pick up the slot pointer for the dynamic operand (the EXACT-size
// buffer the publisher allocated, not the upper-bound DPS init).
//
// The dim_slots attribute is NOT expected here because the view
// reshape breaks the original (result, dim) -> DimSpec mapping; the
// transpose lowering still reads the expanded shape from the
// descriptor, but pointer redirection alone is enough to read
// correct DATA from the slot buffer. Dim-slot threading through
// view ops is out-of-scope for Phase 2 (Phase 3 coalescing handles
// the canonicalization side).
//
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.nonzero
// CHECK:         memref.expand_shape
// CHECK:         hip.transpose
// CHECK-SAME:    hipdnn.input_slot_buffers = array<i32: -1, 0, -1>
func.func @main_graph(
    %ctx: !hip.context,
    %mask: memref<3x4xi1, 1>,
    %nz_out: memref<2x?xi64, 1>,
    %tr_out: memref<1x?x2xi64, 1>) {
  hip.nonzero(%ctx) ins(%mask : memref<3x4xi1, 1>)
                    outs(%nz_out : memref<2x?xi64, 1>)
                    {input_data_type = 5 : i64,
                     slot_id = 0 : i32,
                     output_dim_specs = [[
                       [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
                       [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
                     ]]}
  %expanded = memref.expand_shape %nz_out [[0, 1], [2]] output_shape [1, 2, 0]
              : memref<2x?xi64, 1> into memref<1x2x?xi64, 1>
  hip.transpose(%ctx) ins(%expanded : memref<1x2x?xi64, 1>)
                      outs(%tr_out : memref<1x?x2xi64, 1>)
                      {perm = [0, 2, 1]}
  return
}
