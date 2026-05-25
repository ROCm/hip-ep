// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// dim-specs-debug.mlir
//
// Covers the analysis-only DimSpec debug passes registered for
// hip-mlir-opt:
//
//   --hip-dump-dim-specs    Pretty-prints every DimSpec the module carries
//                           (per-op output_dim_specs + module-level
//                           hipdnn.output_dim_specs) to stderr in the same
//                           textual form as the EP-side
//                           HIPDNN_EP_DEBUG_SHAPES tracer.
//
//   --hip-verify-dim-specs  Structurally validates the same metadata.
//                           Returns success ("hip-verify-dim-specs: OK")
//                           when every tree is well-formed and every
//                           dynamic result dim has a non-empty DimSpec
//                           entry; signalPassFailure otherwise.
//
// Two RUN lines (one for each pass), exercised against three inputs split
// with split-input-file: Category-A (all-static), Category-B (input-shape
// derived), and Category-C (NonZero -> RuntimeSlot). Both passes are
// analysis-only and never modify the IR; FileCheck matches the stderr
// dump for the dumper and the cumulative "OK" sentinel for the verifier.

// RUN: hip-mlir-opt --split-input-file --onnx-to-hip-pipeline \
// RUN:   --hip-compose-dim-specs --hip-dump-dim-specs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DUMP_A,DUMP_B,DUMP_C

// RUN: hip-mlir-opt --split-input-file --onnx-to-hip-pipeline \
// RUN:   --hip-compose-dim-specs --hip-verify-dim-specs %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=VERIFY

// Category A: identity on a fully-static tensor. Every output dim is a
// Static leaf and the slot counter is 0.
// DUMP_A: === hip-dump-dim-specs ===
// DUMP_A: [module-level hipdnn.output_dim_specs]
// DUMP_A: Output[0] dim[0] = 4
// DUMP_A: Output[0] dim[1] = 8
// DUMP_A: [hipdnn.dyn_dim_slots_count] = 0
// VERIFY: hip-verify-dim-specs: OK
module {
  func.func @main_graph(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"}
        : (tensor<4x8xf32>) -> ()
  }
}

// -----

// Category B: dynamic output dim resolves to an input dim. Composed spec
// has InputDim leaves; slot counter is still 0. Static middle dim renders
// as a constant.
// DUMP_B: === hip-dump-dim-specs ===
// DUMP_B: [module-level hipdnn.output_dim_specs]
// DUMP_B: Output[0] dim[0] = arg[0].shape[0]
// DUMP_B: Output[0] dim[1] = 3
// DUMP_B: Output[0] dim[2] = arg[0].shape[2]
// DUMP_B: [hipdnn.dyn_dim_slots_count] = 0
// VERIFY: hip-verify-dim-specs: OK
module {
  func.func @main_graph(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"}
        : (tensor<?x3x?xf32>) -> ()
  }
}

// -----

// Category C: NonZero output dim 1 is data-dependent. The composed spec
// must contain a RuntimeSlot leaf, the per-op output_dim_specs on
// hip.nonzero must also surface, and the slot count must be at least 1.
// DUMP_C: === hip-dump-dim-specs ===
// DUMP_C: [per-op DimSpecs]
// DUMP_C: hip.nonzero
// DUMP_C:   output_dim_specs[result][dim]:
// DUMP_C: [module-level hipdnn.output_dim_specs]
// DUMP_C: Output[0] dim[0] = 1
// DUMP_C: Output[0] dim[1] = slot[0]
// DUMP_C: [hipdnn.dyn_dim_slots_count] = 1
// VERIFY: hip-verify-dim-specs: OK
module {
  func.func @main_graph(%arg0: tensor<4xi64>) -> tensor<1x?xi64> {
    %r = "onnx.NonZero"(%arg0) {onnx_node_name = "/NonZero"}
        : (tensor<4xi64>) -> tensor<1x?xi64>
    "onnx.Return"(%r) {onnx_node_name = "/Return"}
        : (tensor<1x?xi64>) -> ()
  }
}
