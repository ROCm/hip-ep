// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// test_slot_lifetime_coalesce_footprint.mlir
//
// Phase 3 end-to-end footprint regression for the slot-buffer-coalesce
// design (docs/design/slot-buffer-coalesce.md).
//
// Graph shape -- mirrors the POSITIVE case in
// test/lit/Transforms/slot-lifetime-coalesce.mlir at the ONNX-source level:
//
//   * ONE NonZero publishes slot 0 (`N`, the dynamic dim).
//   * TWO downstream Transpose chains BOTH read the same NonZero result
//     and each reserves its own propagator output slot (slots 1 and 2)
//     whose DimSpec for the dynamic dim is `RuntimeSlot(0)` -- identical
//     canonical bytes.
//   * Each intermediate Transpose feeds a final Transpose that
//     materialises a static-shape output (`tensor<2x3xi64>`). This is
//     the key shape requirement: the propagator slots MUST NOT be
//     output-bound for the coalescer to merge them. If they fed
//     `onnx.Return` directly, both would carry `[def, +inf)` lifetimes
//     and Phase 3 would (correctly) leave them distinct -- see the
//     "two output-bound slots cannot share" NEGATIVE case in the
//     unit test for the same reason.
//
// Lifetimes after Phase 3:
//   * slot 0 (NonZero result, consumed by both intermediate Transposes
//     and -- via shape rewiring -- by both final Transposes) is
//     output-bound only insofar as its DimSpec leaves appear in
//     `hipdnn.output_dim_specs`; the buffer itself is intermediate.
//   * slot 1 (first intermediate Transpose) dies at first final
//     Transpose.
//   * slot 2 (second intermediate Transpose) is defined AFTER slot 1
//     dies -- non-overlapping.
//   * Slots 1 and 2 have identical canonical DimSpec bytes
//     (`[RuntimeSlot(0), Static(2)]` in some order), so the coalescer
//     merges slot 2 into slot 1 and renumbers contiguously.
//
// Post-coalesce: `hipdnn.dyn_dim_slots_count = 2` (slot 0 for the
// NonZero publisher; one merged slot for both intermediate Transposes).

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-DAG: hipdnn.dyn_dim_slots_count = 2 : i32
// CHECK-DAG: hipdnn.pool_size = {{[0-9]+}} : i64
// CHECK-DAG: llvm.func @wrap_nonzero
// CHECK-DAG: llvm.func @wrap_transpose

module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"})
      -> (tensor<2x3xi64> {onnx.name = "out_a"},
          tensor<2x3xi64> {onnx.name = "out_b"}) {
    %nz = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero"}
        : (tensor<3x4xi1>) -> tensor<2x?xi64>
    %t1 = "onnx.Transpose"(%nz) {onnx_node_name = "transpose_a", perm = [1, 0]}
        : (tensor<2x?xi64>) -> tensor<?x2xi64>
    %fin1 = "onnx.Transpose"(%t1) {onnx_node_name = "final_a", perm = [1, 0]}
        : (tensor<?x2xi64>) -> tensor<2x3xi64>
    %t2 = "onnx.Transpose"(%nz) {onnx_node_name = "transpose_b", perm = [1, 0]}
        : (tensor<2x?xi64>) -> tensor<?x2xi64>
    %fin2 = "onnx.Transpose"(%t2) {onnx_node_name = "final_b", perm = [1, 0]}
        : (tensor<?x2xi64>) -> tensor<2x3xi64>
    "onnx.Return"(%fin1, %fin2) : (tensor<2x3xi64>, tensor<2x3xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
