// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// dim-specs-categories.mlir
//
// End-to-end pipeline check for the operand-provenance-aware DimSpec
// emission added with the general op-level dynamic output shapes work.
// Three small graphs exercise the three categories that the EP host-side
// resolver dispatches on:
//
//   Category A: output dim is a compile-time static value. Encoded as
//   `Static(N)` in the DimSpec tree; no slot, no input-arg indirection.
//
//   Category B: output dim is a function of one or more EP-input shape
//   or i64-value reads. Encoded as an arithmetic tree bottoming out in
//   `InputDim` / `InputValueI64` leaves; resolved pre-compute by the EP.
//   Still no slot needed.
//
//   Category C: output dim is data-dependent at runtime. Encoded as
//   `RuntimeSlot(K)`; the producing op publishes the dim value (and a
//   GPU buffer) into slot K from inside its wrap_* runtime function; the
//   EP reads slot K post-compute. Always emits a non-zero
//   `hipdnn.dyn_dim_slots_count`.
//
// The three sub-tests below use mlir-opt's split-input-file mode so that
// each separator starts a fresh top-level module with its own
// `@main_graph` (required by `--convert-onnx-to-hip`'s metadata
// generator). They run the full ONNX->HIP -> ComposeDimSpecs pipeline
// and check that the lowered IR contains the expected DimSpec node-kind
// for each output dim. ComposeDimSpecs needs the IR to be bufferized to
// out-params first (it walks the trailing out-param block args of
// `@main_graph`), so we run `--onnx-to-hip-pipeline` and then the
// compose pass. Each sub-test selects a distinct FileCheck prefix
// (CAT_A / CAT_B / CAT_C) so the per-category CHECK rules apply to the
// matching split output only.

// RUN: hip-mlir-opt --split-input-file --onnx-to-hip-pipeline --hip-compose-dim-specs %s | FileCheck %s --check-prefixes=CAT_A,CAT_B,CAT_C

// Category A: every output dim is a compile-time constant. Identity on a
// fully-static tensor -> all dims encode as `Static`.
// `hipdnn.dyn_dim_slots_count` is 0 (no Category-C producer in the
// graph).
// CAT_A: hipdnn.dyn_dim_slots_count = 0 : i32
// CAT_A-NOT: hipdnn.next_dyn_slot_id
module {
  func.func @main_graph(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"}
        : (tensor<4x8xf32>) -> ()
  }
}

// -----

// Category B: dynamic output dim resolves to an EP-input dimension.
// Identity on a partially-dynamic tensor -> the composed dim spec is an
// `InputDim` leaf, NOT a `RuntimeSlot`. Slots counter is still 0.
// CAT_B: hipdnn.dyn_dim_slots_count = 0 : i32
// CAT_B-NOT: hipdnn.next_dyn_slot_id
module {
  func.func @main_graph(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"}
        : (tensor<?x3x?xf32>) -> ()
  }
}

// -----

// Category C: NonZero's output dim 1 is genuinely data-dependent. The
// composed spec must contain a `RuntimeSlot` leaf, and the module-level
// slots counter must be at least 1.
// CAT_C: hipdnn.dyn_dim_slots_count = 1 : i32
// CAT_C: hipdnn.next_dyn_slot_id = 1 : i32
// CAT_C: hip.nonzero
// CAT_C-SAME: slot_id = 0 : i32
module {
  func.func @main_graph(%arg0: tensor<4xi64>) -> tensor<1x?xi64> {
    %r = "onnx.NonZero"(%arg0) {onnx_node_name = "/NonZero"}
        : (tensor<4xi64>) -> tensor<1x?xi64>
    "onnx.Return"(%r) {onnx_node_name = "/Return"}
        : (tensor<1x?xi64>) -> ()
  }
}
