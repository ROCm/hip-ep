// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// reserve-propagator-slots.mlir
//
// Phase 2.3 of the slot-buffer-coalescing initiative
// (docs/design/slot-buffer-coalesce.md). The pass walks @main_graph,
// finds every Hip dialect op whose result has a dynamic dim that
// transitively depends on a Cat-C RuntimeSlot leaf, and attaches the
// `hipdnn.output_slot_ids` array attribute (i32[numResults][rank])
// with a fresh slot id per dynamic-and-slot-dependent dim, -1
// elsewhere. Cat-C publishers (those already carrying slot_id /
// slot_ids / hipdnn.output_slot_ids) are left untouched. Static ops
// and ops whose dynamic dims resolve to non-slot DimSpecs (Cat-B / A)
// are also skipped.
//
// The DimSpec encoding is the 8-field i64 array form:
//   [kind, value, in_idx, dim_idx, flat_off, slot_id, lhs, rhs]
// with kinds Static=0, InputDim=1, InputValueI64=2, RuntimeSlot=3.

// RUN: hip-mlir-opt --split-input-file --hip-reserve-propagator-slots %s | FileCheck %s

// === RESERVE for a translucent propagator ===
//
// NonZero publishes slot 0; Transpose's dim 0 inherits via the perm.
// The pass must annotate Transpose with output_slot_ids = [[1, -1]]
// (dim 0 is dynamic and slot-bound; dim 1 is static R=2). The
// module-level slot counter advances from 1 to 2.
//
// CHECK: module attributes {hipdnn.next_dyn_slot_id = 2 : i32}
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.nonzero
// CHECK-SAME:    slot_id = 0
// CHECK:         hip.transpose
// CHECK-SAME:    hipdnn.output_slot_ids = {{\[}}array<i32: 1, -1>]
module attributes {hipdnn.next_dyn_slot_id = 1 : i32} {
  func.func @main_graph(%ctx: !hip.context, %input: tensor<3x4xi1>) -> tensor<?x2xi64> {
    %ub = arith.constant 12 : index
    %nz_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz = hip.nonzero(%ctx) ins(%input : tensor<3x4xi1>) outs(%nz_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    %t_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t_init : tensor<?x2xi64>) {
      perm = [1, 0]
    } : tensor<?x2xi64>
    return %t : tensor<?x2xi64>
  }
}

// -----

// === SKIP a Cat-C publisher (already has slot_id) ===
//
// The pass leaves NonZero alone -- it already has slot_id = 0. No
// hipdnn.output_slot_ids should appear on it.
//
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.nonzero
// CHECK-NOT:     hipdnn.output_slot_ids
module attributes {hipdnn.next_dyn_slot_id = 1 : i32} {
  func.func @main_graph(%ctx: !hip.context, %input: tensor<3x4xi1>) -> tensor<2x?xi64> {
    %ub = arith.constant 12 : index
    %nz_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz = hip.nonzero(%ctx) ins(%input : tensor<3x4xi1>) outs(%nz_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    return %nz : tensor<2x?xi64>
  }
}

// -----

// === SKIP an op with NO dynamic dim ===
//
// Transpose on a fully static input -- no dynamic output dim, so
// no slot to reserve. The pass must NOT attach hipdnn.output_slot_ids.
//
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.transpose
// CHECK-NOT:     hipdnn.output_slot_ids
module {
  func.func @main_graph(%ctx: !hip.context, %in: tensor<3x4xf32>) -> tensor<4x3xf32> {
    %init = tensor.empty() : tensor<4x3xf32>
    %t = hip.transpose(%ctx) ins(%in : tensor<3x4xf32>) outs(%init : tensor<4x3xf32>) {
      perm = [1, 0]
    } : tensor<4x3xf32>
    return %t : tensor<4x3xf32>
  }
}

// -----

// === RESERVE slots for a Range -> Tile chain (vision.onnx pattern) ===
//
// Mirrors the vision.onnx Tile-of-slot pattern (Cat-C-downstream
// Tile): the input dim comes from a Cat-C Range publisher (slot 0)
// and tiling repeats it `R` times so the output dim is
// `Mul(RuntimeSlot(0), Static(R))`. ReservePropagatorSlotsPass must
// allocate a fresh slot for the propagator dim because the DimSpec
// still contains a RuntimeSlot leaf. (The Mul arithmetic is the
// in-DLL compound evaluator's domain at consumer-read time, but the
// reservation works on the leaf-containment predicate only.)
//
// We model the Range with a fixed `slot_id = 0` and Tile downstream
// as a hand-rolled hip op. Tile's per-result output_dim_specs is
// attached so getResultDimSpec finds a RuntimeSlot-containing tree.
//
// CHECK: module attributes {hipdnn.next_dyn_slot_id = 2 : i32}
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.range
// CHECK-SAME:    slot_id = 0
// CHECK:         hip.tile
// CHECK-SAME:    hipdnn.output_slot_ids = {{\[}}array<i32: 1>]
module attributes {hipdnn.next_dyn_slot_id = 1 : i32} {
  func.func @main_graph(
      %ctx: !hip.context,
      %start: tensor<i64>, %limit: tensor<i64>, %delta: tensor<i64>,
      %repeats: tensor<1xi64>) -> tensor<?xi64> {
    %ub = arith.constant 1024 : index
    %r_init = tensor.empty(%ub) : tensor<?xi64>
    %r = hip.range(%ctx) ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
                         outs(%r_init : tensor<?xi64>) {
      slot_id = 0 : i32,
      output_dim_specs = [[[array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]]]
    } : tensor<?xi64>
    %t_init = tensor.empty(%ub) : tensor<?xi64>
    // Tile's output_dim_specs has a 3-node tree: Mul(RuntimeSlot(0), Static(2))
    // -- node 0 is the binary Mul (lhs=1, rhs=2), node 1 the RuntimeSlot leaf,
    // node 2 the Static leaf.
    %t = hip.tile(%ctx) ins(%r, %repeats : tensor<?xi64>, tensor<1xi64>)
                        outs(%t_init : tensor<?xi64>) {
      output_dim_specs = [[[
        array<i64: 4, 0, 0, 0, 0, 0, 1, 2>,
        array<i64: 3, 0, 0, 0, 0, 0, 0, 0>,
        array<i64: 0, 2, 0, 0, 0, 0, 0, 0>
      ]]]
    } : tensor<?xi64>
    return %t : tensor<?xi64>
  }
}
