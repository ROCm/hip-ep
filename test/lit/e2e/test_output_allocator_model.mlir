// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: output-allocator output handling, end-to-end through
// --hipdnn-pipeline on a single-Sigmoid model.
//
// The output-allocator ABI is the only mode:
//   - graph output is allocated in-graph: hip.alloc_output lowers to a
//     llvm.call @hipdnn_ep_alloc_output (the EP output-allocator callback).
//   - inference_compute has the 2-arg (state, inputs) ABI -- no outputs span.
//   - the output buffer is EP-owned and written in place (no output staging).
// ============================================================================

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// In-graph output allocation + 2-arg compute ABI.
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: llvm.func @inference_compute(%{{[a-zA-Z0-9_]+}}: !llvm.ptr, %{{[a-zA-Z0-9_]+}}: !llvm.ptr) -> i32

module {
  func.func @main_graph(%arg0: tensor<4x8xf16> {onnx.name = "input"}) -> (tensor<4x8xf16> {onnx.name = "output"}) {
    %0 = "onnx.Sigmoid"(%arg0) {onnx_node_name = "/Sigmoid"} : (tensor<4x8xf16>) -> tensor<4x8xf16>
    "onnx.Return"(%0) : (tensor<4x8xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
