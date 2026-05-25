// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-annotate-input-dim-slots.
//
// The pass walks every op in @main_graph. For each operand whose
// producer is a slot-publishing Category-C op (carries `slot_id`)
// and whose `output_dim_specs` contain a `RuntimeSlot` leaf, it
// attaches two attributes on the consumer:
//
//   hipdnn.input_dim_slots     -- ArrayAttr of [dim_idx, slot_id]
//                                 pairs per operand; consumer SHAPE
//                                 rewiring uses this to read the
//                                 published dim via
//                                 `hipdnn_ep_state_read_dim`.
//
//   hipdnn.input_slot_buffers  -- DenseI32ArrayAttr (one i32 per
//                                 operand) recording the slot id of
//                                 the immediate producer when that
//                                 producer is itself a slot
//                                 publisher; consumer POINTER
//                                 rewiring uses this to read the
//                                 published exact-size buffer via
//                                 `hipdnn_ep_state_peek_buffer`.
//                                 -1 means "use descriptor pointer
//                                 as usual" (e.g. operand is a
//                                 propagator output that wrote into
//                                 its own upper-bound DPS init).
//
// Consumer ops that resolve no slot get NEITHER attribute. Slot
// publishers themselves (carrying `slot_id`) are explicitly skipped
// so they never self-annotate.
//===----------------------------------------------------------------------===//

// The pass only walks `@main_graph` (the canonical entry-point func).
// We exercise each scenario as its own RUN invocation against a tiny
// `@main_graph` body, and CHECK / CHECK-NOT in separate FileCheck
// runs. Splitting the file (`--check-prefix=`) keeps every scenario's
// expected IR independent.

// RUN: hip-mlir-opt --hip-compose-dim-specs --hip-annotate-input-dim-slots %s --split-input-file | FileCheck %s

// ===== Direct consumer of a slot publisher (hip.transpose <- hip.nonzero) =====
//
// hip.nonzero publishes its dynamic output dim to slot 0. The
// transpose's `input` operand (operand 1, after the ctx at operand 0)
// has its dim 1 = slot[0]. The transpose's `output` operand (operand
// 2) is a separate buffer that the transpose itself writes into, so
// no slot annotation is recorded for it.
//
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.nonzero
// CHECK-NOT:     hipdnn.input_dim_slots
// CHECK:         hip.transpose
// CHECK-SAME:    hipdnn.input_dim_slots = {{\[}}[], {{\[}}array<i32: 1, 0>], []]
// CHECK-SAME:    hipdnn.input_slot_buffers = array<i32: -1, 0, -1>
// Manually attach `output_dim_specs` on hip.nonzero to mimic what the
// onnx-to-hip conversion would emit (Strategy 1 in `getResultDimSpec`
// — per-op attribute wins). Encoding is an ArrayAttr of
// DenseI64ArrayAttr fields: [kind, value, in_idx, dim_idx, flat_off,
// slot_id, lhs, rhs] (kEncodingArity=8). Kinds: Static=0,
// RuntimeSlot=3. Output[0] is rank=2 = static-2; output[1] is the
// data-dependent count = slot[0].
func.func @main_graph(
    %ctx: !hip.context,
    %mask: memref<3x4xi1, 1>,
    %nz_out: memref<2x?xi64, 1>,
    %tr_out: memref<?x2xi64, 1>) {
  hip.nonzero(%ctx) ins(%mask : memref<3x4xi1, 1>)
                    outs(%nz_out : memref<2x?xi64, 1>)
                    {input_data_type = 5 : i64,
                     slot_id = 0 : i32,
                     output_dim_specs = [[
                       [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
                       [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
                     ]]}
  hip.transpose(%ctx) ins(%nz_out : memref<2x?xi64, 1>)
                      outs(%tr_out : memref<?x2xi64, 1>)
                      {perm = [1, 0]}
  return
}

// -----

// ===== All-static operands get no annotation =====
//
// A transpose whose input has only static dims has no RuntimeSlot
// in its operand's DimSpec, so neither attribute is attached.
//
// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.transpose
// CHECK-NOT:     hipdnn.input_dim_slots
// CHECK-NOT:     hipdnn.input_slot_buffers
func.func @main_graph(
    %ctx: !hip.context,
    %a: memref<2x3xi64, 1>,
    %b: memref<3x2xi64, 1>) {
  hip.transpose(%ctx) ins(%a : memref<2x3xi64, 1>)
                      outs(%b : memref<3x2xi64, 1>)
                      {perm = [1, 0]}
  return
}
