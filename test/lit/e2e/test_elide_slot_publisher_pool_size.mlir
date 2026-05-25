// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// test_elide_slot_publisher_pool_size.mlir
//
// Phase 1 pool-size regression assertion for the slot-buffer-coalescing
// initiative (docs/design/slot-buffer-coalesce.md).
//
// A graph whose ONLY allocator is `onnx.NonZero` (a Cat-C publisher
// that allocates+publishes its own exact-size buffer at runtime). With
// the ElideSlotPublisherAllocsPass:
//
//   * the bufferize-materialised `memref.alloc(%upperBound)` for the
//     NonZero result has its dyn-size operand replaced by
//     `arith.constant 0`;
//   * `--hip-pool-allocs` then packs it as a 0-byte dynamic bucket
//     whose contribution to `hipdnn.pool_size` is exactly zero.
//
// FileCheck asserts:
//   (a) the module-level `hipdnn.pool_size` attribute = 0 (the
//       publisher's UB buffer was elided), AND
//   (b) `wrap_nonzero` is still called (the runtime publisher path is
//       preserved -- only the DPS-init reservation is gone).

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-DAG: hipdnn.pool_size = 0 : i64
// CHECK-DAG: hipdnn.dyn_dim_slots_count = 1
// CHECK-DAG: llvm.func @wrap_nonzero
module {
  func.func @main_graph(%arg0: tensor<3x4xi1> {onnx.name = "input"})
      -> (tensor<2x?xi64> {onnx.name = "output"}) {
    %0 = "onnx.NonZero"(%arg0) {onnx_node_name = "nonzero_node"}
        : (tensor<3x4xi1>) -> tensor<2x?xi64>
    "onnx.Return"(%0) : (tensor<2x?xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
