// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// identity-propagator-rebind.mlir
//
// Phase 4 of the slot-buffer-coalesce design
// (docs/design/slot-buffer-coalesce.md). The pass walks `@main_graph`,
// finds ops with a registered identity predicate
// (`shape_interface::isIdentityOp`), erases them (RAUW result with
// input), and substitutes the upstream input slot for the propagator's
// own output slot wherever it appears (publisher attrs, consumer
// attrs, RuntimeSlot leaves, module-level metadata).

// RUN: hip-mlir-opt --split-input-file --hip-identity-propagator-rebind %s | FileCheck %s

// === POSITIVE: identity-perm transpose elided + slot rebound ===
//
// `hip.nonzero` publishes slot 0 (the N dim of its result). An
// identity `hip.transpose` (perm = [0, 1]) downstream reserves slot 1
// for the same N dim. With the identity predicate true, the pass
// erases the transpose, RAUWs its result with the nonzero output, and
// rebinds the consumer's `hipdnn.input_slot_buffers` reference from
// slot 1 to slot 0.
//
// CHECK-LABEL: func.func @identity_transpose
// CHECK:        hip.nonzero
// CHECK-SAME:   slot_id = 0
// CHECK-NOT:    hip.transpose
// CHECK:        hip.scatter_nd
// CHECK-SAME:   hipdnn.input_slot_buffers = array<i32: -1, 0
module attributes {hipdnn.dyn_dim_slots_count = 2 : i32, hipdnn.next_dyn_slot_id = 2 : i32} {
  func.func @identity_transpose(%ctx: !hip.context, %x: tensor<3x4xi1>,
                                %data: tensor<8x8xf32>, %upd: tensor<3x4xf32>)
      -> tensor<8x8xf32> {
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
    %t_init = tensor.empty(%ub) : tensor<2x?xi64>
    %t = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t_init : tensor<2x?xi64>) {
      perm = [0, 1],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: -1, 1>],
      output_dim_specs = [[
        [array<i64: 0, 2, 0, 0, 0, -1, 0, 0>],
        [array<i64: 3, 0, 0, 0, 0, 0, 1, 0>]
      ]]
    } : tensor<2x?xi64>
    %sn_init = tensor.empty() : tensor<8x8xf32>
    %sn = hip.scatter_nd(%ctx) ins(%data, %t, %upd : tensor<8x8xf32>, tensor<2x?xi64>, tensor<3x4xf32>) outs(%sn_init : tensor<8x8xf32>) {
      hipdnn.input_slot_buffers = array<i32: -1, 1, -1>
    } : tensor<8x8xf32>
    return %sn : tensor<8x8xf32>
  }
}

// -----

// === POSITIVE: non-identity perm transpose preserved ===
//
// Same shape as the above but with a NON-identity permutation
// `perm = [1, 0]`. The predicate returns false; the pass must leave
// the transpose intact.
//
// CHECK-LABEL: func.func @non_identity_transpose
// CHECK:        hip.nonzero
// CHECK:        hip.transpose
// CHECK-SAME:   perm = [1, 0]
module attributes {hipdnn.dyn_dim_slots_count = 2 : i32, hipdnn.next_dyn_slot_id = 2 : i32} {
  func.func @non_identity_transpose(%ctx: !hip.context, %x: tensor<3x4xi1>)
      -> tensor<?x2xi64> {
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
    %t_init = tensor.empty(%ub) : tensor<?x2xi64>
    %t = hip.transpose(%ctx) ins(%nz : tensor<2x?xi64>) outs(%t_init : tensor<?x2xi64>) {
      perm = [1, 0],
      hipdnn.input_slot_buffers = array<i32: -1, 0>,
      hipdnn.output_slot_ids = [array<i32: 1, -1>]
    } : tensor<?x2xi64>
    return %t : tensor<?x2xi64>
  }
}

// -----

// === POSITIVE: same-dtype cast elided ===
//
// `hip.cast` between two f32 tensors of identical shape is a runtime
// no-op. The predicate is true and the cast is erased.
//
// CHECK-LABEL: func.func @identity_cast
// CHECK-NOT:    hip.cast
// CHECK:        return
module {
  func.func @identity_cast(%ctx: !hip.context, %x: tensor<3x4xf32>)
      -> tensor<3x4xf32> {
    %init = tensor.empty() : tensor<3x4xf32>
    %c = hip.cast(%ctx) ins(%x : tensor<3x4xf32>) outs(%init : tensor<3x4xf32>) {
      to = 1 : i64
    } : tensor<3x4xf32>
    return %c : tensor<3x4xf32>
  }
}

// -----

// === POSITIVE: cross-dtype cast preserved ===
//
// CHECK-LABEL: func.func @non_identity_cast
// CHECK:        hip.cast
// CHECK:        return
module {
  func.func @non_identity_cast(%ctx: !hip.context, %x: tensor<3x4xf32>)
      -> tensor<3x4xf16> {
    %init = tensor.empty() : tensor<3x4xf16>
    %c = hip.cast(%ctx) ins(%x : tensor<3x4xf32>) outs(%init : tensor<3x4xf16>) {
      to = 10 : i64
    } : tensor<3x4xf16>
    return %c : tensor<3x4xf16>
  }
}

// -----

// === POSITIVE: same-shape expand elided ===
//
// `hip.expand` to a target shape that equals the input shape on every
// dim is a runtime no-op.
//
// CHECK-LABEL: func.func @identity_expand
// CHECK-NOT:    hip.expand
// CHECK:        return
module {
  func.func @identity_expand(%ctx: !hip.context, %x: tensor<2x3xf32>,
                             %shape: tensor<2xi64>)
      -> tensor<2x3xf32> {
    %init = tensor.empty() : tensor<2x3xf32>
    %e = hip.expand(%ctx) ins(%x, %shape : tensor<2x3xf32>, tensor<2xi64>) outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>
    return %e : tensor<2x3xf32>
  }
}

// -----

// === POSITIVE: full-range slice elided ===
//
// `hip.slice` with provable starts=[0,0], ends=[3,4] (== full dim),
// steps=[1,1] is a runtime no-op and the op is erased.
//
// CHECK-LABEL: func.func @identity_slice
// CHECK-NOT:    hip.slice
// CHECK:        return
module {
  func.func @identity_slice(%ctx: !hip.context, %x: tensor<3x4xf32>)
      -> tensor<3x4xf32> {
    %starts = arith.constant dense<[0, 0]> : tensor<2xi64>
    %ends = arith.constant dense<[3, 4]> : tensor<2xi64>
    %steps = arith.constant dense<[1, 1]> : tensor<2xi64>
    %axes = arith.constant dense<[0, 1]> : tensor<2xi64>
    %init = tensor.empty() : tensor<3x4xf32>
    %s = hip.slice(%ctx) ins(%x, %starts, %ends : tensor<3x4xf32>, tensor<2xi64>, tensor<2xi64>) axes(%axes : tensor<2xi64>) steps(%steps : tensor<2xi64>) outs(%init : tensor<3x4xf32>) : tensor<3x4xf32>
    return %s : tensor<3x4xf32>
  }
}

// -----

// === NEGATIVE: reverse slice with same output shape preserved ===
//
// `hip.slice` with start=2, end=-4 (== -1 after normalisation), step=-1
// over axis 0 of a [3xf32] reverses the 3 elements. The output type is
// `tensor<3xf32>` -- same as the input -- but the data permutes. The
// pass MUST NOT elide this op (regression for the bug where same-shape
// shortcut alone treated all slices as identity).
//
// CHECK-LABEL: func.func @reverse_slice_same_shape
// CHECK:        hip.slice
// CHECK:        return
module {
  func.func @reverse_slice_same_shape(%ctx: !hip.context, %x: tensor<3xf32>)
      -> tensor<3xf32> {
    %starts = arith.constant dense<[2]> : tensor<1xi64>
    %ends = arith.constant dense<[-4]> : tensor<1xi64>
    %axes = arith.constant dense<[0]> : tensor<1xi64>
    %steps = arith.constant dense<[-1]> : tensor<1xi64>
    %init = tensor.empty() : tensor<3xf32>
    %s = hip.slice(%ctx) ins(%x, %starts, %ends : tensor<3xf32>, tensor<1xi64>, tensor<1xi64>) axes(%axes : tensor<1xi64>) steps(%steps : tensor<1xi64>) outs(%init : tensor<3xf32>) : tensor<3xf32>
    return %s : tensor<3xf32>
  }
}

// -----

// === NEGATIVE: runtime-value starts/ends preserved ===
//
// When starts/ends arrive as func args (runtime values), the pass
// cannot prove the slice is full-range and MUST leave the op intact
// even when the output type happens to match the input type. This
// is the safety contract that prevents a runtime-controlled reverse
// or sub-range slice from being silently elided.
//
// CHECK-LABEL: func.func @runtime_indices_slice
// CHECK:        hip.slice
// CHECK:        return
module {
  func.func @runtime_indices_slice(%ctx: !hip.context, %x: tensor<3x4xf32>,
                                   %starts: tensor<2xi64>, %ends: tensor<2xi64>)
      -> tensor<3x4xf32> {
    %init = tensor.empty() : tensor<3x4xf32>
    %s = hip.slice(%ctx) ins(%x, %starts, %ends : tensor<3x4xf32>, tensor<2xi64>, tensor<2xi64>) outs(%init : tensor<3x4xf32>) : tensor<3x4xf32>
    return %s : tensor<3x4xf32>
  }
}

// -----

// === POSITIVE: tile with all-ones repeats elided ===
//
// `hip.tile` whose output dims all match the input dims must have
// `repeats == [1, 1, ..., 1]` -- it's a runtime no-op.
//
// CHECK-LABEL: func.func @identity_tile
// CHECK-NOT:    hip.tile
// CHECK:        return
module {
  func.func @identity_tile(%ctx: !hip.context, %x: tensor<3x4xf32>,
                           %r: tensor<2xi64>)
      -> tensor<3x4xf32> {
    %init = tensor.empty() : tensor<3x4xf32>
    %t = hip.tile(%ctx) ins(%x, %r : tensor<3x4xf32>, tensor<2xi64>) outs(%init : tensor<3x4xf32>) : tensor<3x4xf32>
    return %t : tensor<3x4xf32>
  }
}

// -----

// === POSITIVE: reduce_sum with noop_with_empty_axes elided ===
//
// `hip.reduce_sum` with `noop_with_empty_axes = 1` and a shape that
// matches the input dim-for-dim must be a runtime no-op (the axes
// tensor is empty and the convention says to skip the reduction).
//
// CHECK-LABEL: func.func @identity_reduce_sum
// CHECK-NOT:    hip.reduce_sum
// CHECK:        return
module {
  func.func @identity_reduce_sum(%ctx: !hip.context, %x: tensor<3x4xi64>,
                                 %axes: tensor<0xi64>)
      -> tensor<3x4xi64> {
    %init = tensor.empty() : tensor<3x4xi64>
    %r = hip.reduce_sum(%ctx) ins(%x, %axes : tensor<3x4xi64>, tensor<0xi64>) outs(%init : tensor<3x4xi64>) {
      keepdims = 1 : i64,
      noop_with_empty_axes = 1 : i64
    } : tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }
}

// -----

// === NEGATIVE: reduce_sum WITHOUT noop_with_empty_axes preserved ===
//
// Without the `noop_with_empty_axes = 1` flag, an empty `axes` tensor
// means "reduce ALL axes" by ONNX convention -- the op is NOT a
// no-op. Result shape happens to match the input here because we keep
// dims, but the values change. The pass must leave it intact.
//
// CHECK-LABEL: func.func @non_identity_reduce_sum
// CHECK:        hip.reduce_sum
// CHECK:        return
module {
  func.func @non_identity_reduce_sum(%ctx: !hip.context, %x: tensor<3x4xi64>,
                                     %axes: tensor<0xi64>)
      -> tensor<3x4xi64> {
    %init = tensor.empty() : tensor<3x4xi64>
    %r = hip.reduce_sum(%ctx) ins(%x, %axes : tensor<3x4xi64>, tensor<0xi64>) outs(%init : tensor<3x4xi64>) {
      keepdims = 1 : i64,
      noop_with_empty_axes = 0 : i64
    } : tensor<3x4xi64>
    return %r : tensor<3x4xi64>
  }
}

// -----

// === DISABLED FLAG: with disable-identity-rebind=true, no rewrites ===
//
// RUN: hip-mlir-opt --split-input-file --hip-identity-propagator-rebind='disable-identity-rebind=true' %s | FileCheck %s --check-prefix=DISABLED
//
// DISABLED-LABEL: func.func @disabled_test
// DISABLED:        hip.cast
// DISABLED:        return
module {
  func.func @disabled_test(%ctx: !hip.context, %x: tensor<3x4xf32>)
      -> tensor<3x4xf32> {
    %init = tensor.empty() : tensor<3x4xf32>
    %c = hip.cast(%ctx) ins(%x : tensor<3x4xf32>) outs(%init : tensor<3x4xf32>) {
      to = 1 : i64
    } : tensor<3x4xf32>
    return %c : tensor<3x4xf32>
  }
}
