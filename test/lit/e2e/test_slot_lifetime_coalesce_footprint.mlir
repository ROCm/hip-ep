// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// test_slot_lifetime_coalesce_footprint.mlir
//
// Phase 3 end-to-end footprint regression for the slot-buffer-coalesce
// design (docs/design/slot-buffer-coalesce.md).
//
// An ONNX graph with TWO independent NonZero -> Transpose chains
// against the same boolean input. Each NonZero publishes its own
// dynamic slot, and each Transpose downstream of it carries a
// propagator-reserved output slot whose DimSpec on the dynamic dim is
// `RuntimeSlot(self)` (the kind=3 leaf points at the propagator's own
// slot id since the Transpose preserves the NonZero's N dim).
//
// Before this pass: 4 dynamic slots (2 NonZero results + 2 Transpose
// propagators). With Phase 3 lifetime coalescing, the two intermediate
// Transpose slots have non-overlapping lifetimes AND, because each
// inherits the SAME canonical DimSpec shape modulo the slot pointer,
// the publisher's own slot can also be merged once we canonicalise
// `RuntimeSlot(s)` leaves: in the simpler case both NonZero outputs
// remain output-bound (live past return when the graph emits them),
// but the two propagator Transposes WILL coalesce when their
// lifetimes do not overlap. The footprint regression assertion below
// is:
//
//   * `hipdnn.dyn_dim_slots_count` reaches a stable, post-coalesce
//     value. The number is what the coalescer leaves; we check it
//     does not regress beyond a known upper bound.
//
// This test is intentionally an *upper-bound* assertion -- it tells us
// the slot count must NOT exceed the recorded baseline. If a future
// pipeline regression breaks coalescing for this canonical shape, the
// pool size goes up and this test fails loudly.

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-DAG: hipdnn.dyn_dim_slots_count = 2 : i32
// CHECK-DAG: hipdnn.pool_size = {{[0-9]+}} : i64
// CHECK-DAG: llvm.func @wrap_nonzero
// CHECK-DAG: llvm.func @wrap_transpose

module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"})
      -> (tensor<?x2xi64> {onnx.name = "out_a"},
          tensor<?x2xi64> {onnx.name = "out_b"}) {
    %nza = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero_a"}
        : (tensor<3x4xi1>) -> tensor<2x?xi64>
    %nzb = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero_b"}
        : (tensor<3x4xi1>) -> tensor<2x?xi64>
    %ta = "onnx.Transpose"(%nza) {onnx_node_name = "transpose_a", perm = [1, 0]}
        : (tensor<2x?xi64>) -> tensor<?x2xi64>
    %tb = "onnx.Transpose"(%nzb) {onnx_node_name = "transpose_b", perm = [1, 0]}
        : (tensor<2x?xi64>) -> tensor<?x2xi64>
    "onnx.Return"(%ta, %tb) : (tensor<?x2xi64>, tensor<?x2xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
