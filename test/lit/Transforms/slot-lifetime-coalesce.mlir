// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// slot-lifetime-coalesce.mlir
//
// Phase 3 of the slot-buffer-coalesce design
// (docs/design/slot-buffer-coalesce.md). The pass collects every
// published slot in `@main_graph`, groups by canonical DimSpec bytes,
// first-fit-decreasing-packs by lifetime within each group, picks the
// smallest slot id per bin as the representative, and rewrites every
// slot reference (publisher attrs, consumer attrs, RuntimeSlot leaves
// inside `output_dim_specs`) so the rest of the pipeline sees a
// contiguous remapped slot id range.
//
// DimSpec encoding (8 i64 fields):
//   [kind, value, in_idx, dim_idx, flat_off, slot_id, lhs, rhs]
// kinds: Static=0, InputDim=1, InputValueI64=2, RuntimeSlot=3.

// RUN: hip-mlir-opt --split-input-file --hip-slot-lifetime-coalesce %s | FileCheck %s

// === POSITIVE: two translucent propagator slots coalesce ===
//
// hip.nonzero publishes slot 0 (dim 1 = N from the boolean input).
// Two downstream hip.transpose ops each reserve their own propagator
// output slot (slots 1 and 2) whose DimSpec for the dynamic dim is
// `RuntimeSlot(0)` (they BOTH inherit N from the publisher). After
// fin1 consumes slot 1 (idx 2), slot 1 dies and slot 2 (def idx 3)
// starts -- non-overlapping. Same canonical DimSpec bytes. The
// coalescer rewrites slot 2 -> slot 1 (smaller original id) and
// renumbers contiguously so `hipdnn.dyn_dim_slots_count = 2`. The
// consumer's `hipdnn.input_dim_slots = [[], [array<i32: 0, 2>]]`
// becomes `[[], [array<i32: 0, 1>]]`.
//
// CHECK: module attributes {hipdnn.dyn_dim_slots_count = 2 : i32
// CHECK-LABEL: func.func @main_graph
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 0
// CHECK:        hip.transpose
// CHECK-SAME:   hipdnn.output_slot_ids = {{\[}}array<i32: 1, -1>]
// CHECK:        hip.transpose
// CHECK-SAME:   hipdnn.input_dim_slots = {{\[\[\]}}, {{\[}}array<i32: 0, 1>]]
// CHECK:        hip.transpose
// CHECK-SAME:   hipdnn.output_slot_ids = {{\[}}array<i32: 1, -1>]
// CHECK:        hip.transpose
// CHECK-SAME:   hipdnn.input_dim_slots = {{\[\[\]}}, {{\[}}array<i32: 0, 1>]]
module attributes {hipdnn.dyn_dim_slots_count = 3 : i32, hipdnn.next_dyn_slot_id = 3 : i32} {
  func.func @main_graph(%ctx: !hip.context, %x: tensor<3x4xi1>) -> (tensor<2x3xi64>, tensor<2x3xi64>) {
    %ub = arith.constant 12 : index
    %nz_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nz_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    %t1_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t1 = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t1_init : tensor<?x2xi64>) {
      perm = [1, 0],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: 1, -1>],
      output_dim_specs = [[
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>],
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>]
      ]]
    } : tensor<?x2xi64>
    %fin1_init = tensor.empty() : tensor<2x3xi64>
    %fin1 = hip.transpose(%ctx) ins(%t1 : tensor<?x2xi64>) outs(%fin1_init : tensor<2x3xi64>) {
      perm = [1, 0],
      hipdnn.input_dim_slots = [[], [array<i32: 0, 1>]]
    } : tensor<2x3xi64>
    %t2_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t2 = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t2_init : tensor<?x2xi64>) {
      perm = [1, 0],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: 2, -1>],
      output_dim_specs = [[
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>],
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>]
      ]]
    } : tensor<?x2xi64>
    %fin2_init = tensor.empty() : tensor<2x3xi64>
    %fin2 = hip.transpose(%ctx) ins(%t2 : tensor<?x2xi64>) outs(%fin2_init : tensor<2x3xi64>) {
      perm = [1, 0],
      hipdnn.input_dim_slots = [[], [array<i32: 0, 2>]]
    } : tensor<2x3xi64>
    return %fin1, %fin2 : tensor<2x3xi64>, tensor<2x3xi64>
  }
}

// -----

// === NEGATIVE: distinct canonical DimSpecs keep slots distinct ===
//
// Two NonZero ops publish slot 0 and slot 1. Each carries its own
// `RuntimeSlot(self)` DimSpec on its dynamic dim -- the slot ids
// differ inside the leaves, so the canonical bytes differ. They
// cannot be merged regardless of lifetime.
//
// CHECK: module attributes {hipdnn.dyn_dim_slots_count = 2 : i32
// CHECK-LABEL: func.func @main_graph
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 0
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 1
module attributes {hipdnn.dyn_dim_slots_count = 2 : i32, hipdnn.next_dyn_slot_id = 2 : i32} {
  func.func @main_graph(%ctx: !hip.context, %x: tensor<3x4xi1>) -> (tensor<2x4xi64>, tensor<2x4xi64>) {
    %ub = arith.constant 12 : index
    %nza_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nza = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nza_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    %nzb_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nzb = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nzb_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 1 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 1, 0>]
      ]]
    } : tensor<2x?xi64>
    %ta_init = tensor.empty() : tensor<2x4xi64>
    %ta = hip.transpose(%ctx) ins(%nza : tensor<2x?xi64>) outs(%ta_init : tensor<2x4xi64>) {
      perm = [0, 1],
      hipdnn.input_slot_buffers = array<i32: -1, 0>
    } : tensor<2x4xi64>
    %tb_init = tensor.empty() : tensor<2x4xi64>
    %tb = hip.transpose(%ctx) ins(%nzb : tensor<2x?xi64>) outs(%tb_init : tensor<2x4xi64>) {
      perm = [0, 1],
      hipdnn.input_slot_buffers = array<i32: -1, 1>
    } : tensor<2x4xi64>
    return %ta, %tb : tensor<2x4xi64>, tensor<2x4xi64>
  }
}

// -----

// === NEGATIVE: two output-bound slots cannot share ===
//
// Both NonZero outputs flow directly to func.return, recorded in the
// module's `hipdnn.output_dim_specs`. They are both output-bound
// (lastUse = +inf). Even though their slot ids could in principle be
// renumbered, sharing is forbidden -- both must survive past
// stream-sync for the EP-side resolver to read both. The compose
// attribute also carries distinct slot leaves so the canonical bytes
// differ anyway.
//
// CHECK: module attributes {hipdnn.dyn_dim_slots_count = 2 : i32
// CHECK-LABEL: func.func @main_graph
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 0
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 1
module attributes {
  hipdnn.dyn_dim_slots_count = 2 : i32,
  hipdnn.next_dyn_slot_id = 2 : i32,
  hipdnn.output_dim_specs = [
    [[array<i64: 0, 2, 0, 0, 0, -1, 0, 0>], [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]],
    [[array<i64: 0, 2, 0, 0, 0, -1, 0, 0>], [array<i64: 3, 0, 0, 0, 0, 0, 1, 0>]]
  ]
} {
  func.func @main_graph(%ctx: !hip.context, %x: tensor<3x4xi1>) -> (tensor<2x?xi64>, tensor<2x?xi64>) {
    %ub = arith.constant 12 : index
    %nza_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nza = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nza_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    %nzb_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nzb = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nzb_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 1 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 1, 0>]
      ]]
    } : tensor<2x?xi64>
    return %nza, %nzb : tensor<2x?xi64>, tensor<2x?xi64>
  }
}

// -----

// === STATIC-ONLY backcompat: no slots reserved, pass is a no-op ===
//
// Pure-static graph with no slot publishers anywhere. The pass must
// leave the IR untouched: the printed module attribute list (if
// present) must not gain `hipdnn.dyn_dim_slots_count`, and the
// transpose op must not gain slot-related attributes.
//
// CHECK-LABEL: func.func @static_only
// CHECK-NEXT:   tensor.empty
// CHECK-NEXT:   hip.transpose
// CHECK-SAME:   {perm = [1, 0]}
// CHECK-NEXT:   return
module {
  func.func @static_only(%ctx: !hip.context, %x: tensor<3x4xf32>) -> tensor<4x3xf32> {
    %init = tensor.empty() : tensor<4x3xf32>
    %t = hip.transpose(%ctx) ins(%x : tensor<3x4xf32>) outs(%init : tensor<4x3xf32>) {
      perm = [1, 0]
    } : tensor<4x3xf32>
    return %t : tensor<4x3xf32>
  }
}

// -----

// === disable-coalesce flag preserves source slot ids ===
//
// RUN: hip-mlir-opt --split-input-file --hip-slot-lifetime-coalesce='disable-coalesce=true' %s | FileCheck %s --check-prefix=DISABLED
//
// Three slots in the source IR; with the flag set the pass exits
// early without changing `hipdnn.dyn_dim_slots_count` or any slot id
// on individual ops. Anchor on `@disabled_pass` so we only match the
// module that ships the flag; the four upstream tests do not change
// shape between the default and disabled runs because they're either
// no-coalesce-eligible (canonical bytes differ) or already share
// (test 1's downstream lifetime is fine to coalesce). The
// `disable-coalesce=true` flag is a global pass option that affects
// every split-input module; the on-disk source for the `disabled_pass`
// module has slot_id=2 on the second transpose, and we assert that
// it's preserved.
//
// DISABLED-LABEL: func.func @disabled_pass
// DISABLED:        slot_id = 0
// DISABLED:        hipdnn.output_slot_ids = {{\[}}array<i32: 1, -1>]
// DISABLED:        hipdnn.output_slot_ids = {{\[}}array<i32: 2, -1>]
module attributes {hipdnn.dyn_dim_slots_count = 3 : i32, hipdnn.next_dyn_slot_id = 3 : i32} {
  func.func @disabled_pass(%ctx: !hip.context, %x: tensor<3x4xi1>) -> (tensor<2x3xi64>, tensor<2x3xi64>) {
    %ub = arith.constant 12 : index
    %nz_init = tensor.empty(%ub) : tensor<2x?xi64>
    %nz = hip.nonzero(%ctx) ins(%x : tensor<3x4xi1>) outs(%nz_init : tensor<2x?xi64>) {
      input_data_type = 5 : i64,
      slot_id = 0 : i32,
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>]
      ]]
    } : tensor<2x?xi64>
    %t1_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t1 = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t1_init : tensor<?x2xi64>) {
      perm = [1, 0],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: 1, -1>],
      output_dim_specs = [[
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>],
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>]
      ]]
    } : tensor<?x2xi64>
    %fin1_init = tensor.empty() : tensor<2x3xi64>
    %fin1 = hip.transpose(%ctx) ins(%t1 : tensor<?x2xi64>) outs(%fin1_init : tensor<2x3xi64>) {
      perm = [1, 0],
      hipdnn.input_dim_slots = [[], [array<i32: 0, 1>]]
    } : tensor<2x3xi64>
    %t2_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t2 = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t2_init : tensor<?x2xi64>) {
      perm = [1, 0],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: 2, -1>],
      output_dim_specs = [[
        [array<i64: 3, 0, 0, 0, 0, 0, 0, 0>],
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>]
      ]]
    } : tensor<?x2xi64>
    %fin2_init = tensor.empty() : tensor<2x3xi64>
    %fin2 = hip.transpose(%ctx) ins(%t2 : tensor<?x2xi64>) outs(%fin2_init : tensor<2x3xi64>) {
      perm = [1, 0],
      hipdnn.input_dim_slots = [[], [array<i32: 0, 2>]]
    } : tensor<2x3xi64>
    return %fin1, %fin2 : tensor<2x3xi64>, tensor<2x3xi64>
  }
}
