// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A module of the ggml dialect takes the ggml-to-hip pipeline, of the schema
// the frontend emits; HIPDNN_EP_PIPELINE may name the pipeline too.
//
// RUN: split-file %s %t
// RUN: hip-compiler %t/selected.mlir -o %t/selected.bc
// RUN: llvm-dis %t/selected.bc -o - | FileCheck %s --check-prefix=SELECTED
// RUN: not hip-compiler %t/stale-schema.mlir -o %t/stale-schema.bc 2>&1 | FileCheck %s --check-prefix=STALE-SCHEMA
// RUN: env HIPDNN_EP_PIPELINE=ggml-to-hip-pipeline,hip-to-llvm-pipeline \
// RUN:   hip-compiler %t/selected.mlir -o %t/override.bc
// RUN: llvm-dis %t/override.bc -o - | FileCheck %s --check-prefix=OVERRIDE

// SELECTED: call ptr @hipdnn_ep_alloc_output
// SELECTED: define i32 @inference_compute
// SELECTED-NOT: ggml.add
// OVERRIDE: define i32 @inference_compute

//--- selected.mlir
#dense = #ggml.layout<ne = [4, 1, 1, 1], nb = [4, 16, 16, 16], offset = 0>

module attributes {
  ggml.schema = "hip-ggml-graph-v1"
} {
  func.func @main_graph(
      %lhs: tensor<1x1x1x4xf32, #dense> {
        ggml.io_kind = "runtime",
        ggml.io_name = "runtime.0",
        ggml.source_kind = "input"
      },
      %rhs: tensor<1x1x1x4xf32, #dense> {
        ggml.io_kind = "runtime",
        ggml.io_name = "runtime.1",
        ggml.source_kind = "input"
      }) -> (tensor<1x1x1x4xf32, #dense> {
        ggml.io_kind = "model_output",
        ggml.io_name = "output.0"
      }) {
    %result = "ggml.add"(%lhs, %rhs) :
      (tensor<1x1x1x4xf32, #dense>, tensor<1x1x1x4xf32, #dense>)
        -> tensor<1x1x1x4xf32, #dense>
    return %result : tensor<1x1x1x4xf32, #dense>
  }
}

// STALE-SCHEMA: error: the ggml-to-hip pipeline requires ggml.schema = "hip-ggml-graph-v1"

//--- stale-schema.mlir
module attributes {
  ggml.schema = "hip-ggml-graph-v0"
} {
  func.func @main_graph() {
    return
  }
}
