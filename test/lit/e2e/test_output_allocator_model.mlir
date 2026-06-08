// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// ============================================================================
// TEST: allocator vs classic output handling, end-to-end through
// --hipdnn-pipeline on the same single-Sigmoid model.
//
// Allocator pipeline (use-output-allocator=true):
//   - graph output is allocated in-graph: hip.alloc_output lowers to a
//     llvm.call @hipdnn_ep_alloc_output (the EP output-allocator callback).
//   - inference_compute has the 2-arg (state, inputs) ABI -- no outputs span.
//   - NO output staging calls: prepare_output / finalize_output are skipped
//     (the output buffer is EP-owned and written in place).
//
// Classic pipeline (default):
//   - output is a caller-provided out-param: NO hipdnn_ep_alloc_output.
//   - inference_compute has the 3-arg (state, inputs, outputs) ABI.
//   - prepare_output + finalize_output calls stage the output buffer.
//
// The two modes are pinned against one input so the contrast cannot drift.
// ============================================================================

// RUN: hip-mlir-opt %s --hipdnn-pipeline='use-output-allocator=true' 2>&1 | FileCheck --check-prefix=ALLOC --implicit-check-not="llvm.call @hipdnn_ep_tensor_prepare_output" --implicit-check-not="llvm.call @hipdnn_ep_tensor_finalize_output" %s
// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck --check-prefix=CLASSIC --implicit-check-not=hipdnn_ep_alloc_output %s

// --- Allocator: in-graph output allocation + 2-arg compute ABI. ---
// ALLOC: llvm.call @hipdnn_ep_alloc_output
// ALLOC: llvm.func @inference_compute(%{{[a-zA-Z0-9_]+}}: !llvm.ptr, %{{[a-zA-Z0-9_]+}}: !llvm.ptr) -> i32

// --- Classic: out-param 3-arg compute ABI + output staging calls. ---
// CLASSIC: llvm.func @inference_compute(%{{[a-zA-Z0-9_]+}}: !llvm.ptr, %{{[a-zA-Z0-9_]+}}: !llvm.ptr, %{{[a-zA-Z0-9_]+}}: !llvm.ptr) -> i32
// CLASSIC: llvm.call @hipdnn_ep_tensor_prepare_output
// CLASSIC: llvm.call @hipdnn_ep_tensor_finalize_output

module {
  func.func @main_graph(%arg0: tensor<4x8xf16> {onnx.name = "input"}) -> (tensor<4x8xf16> {onnx.name = "output"}) {
    %0 = "onnx.Sigmoid"(%arg0) {onnx_node_name = "/Sigmoid"} : (tensor<4x8xf16>) -> tensor<4x8xf16>
    "onnx.Return"(%0) : (tensor<4x8xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
