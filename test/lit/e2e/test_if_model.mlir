// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// End-to-end regression guard for onnx.If through the full pipeline.
//
// onnx-if-outline rewrites onnx.If into a tensor-form hip.if (DPS, results
// alias o_init) plus outlined then/else funcs. hip.if MUST have a
// BufferizableOpInterface model registered (HipBufferize.h) or one-shot-
// bufferize aborts with "op was not bufferized: hip.if" — which is silent CPU
// fallback at the EP (a CPU-vs-CPU accuracy test would still pass ~1.0). The
// existing hip.if->llvm LIT starts from the post-bufferize memref form, so it
// cannot catch a missing model; this test runs the whole pipeline so the
// bufferize step is exercised. Reaching the emitted inference_* entry points +
// the hipdnn_ep_run_if driver call proves the graph compiled on the GPU path.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @hipdnn_ep_run_if
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK-NOT: onnx.If
module {
  func.func @main_graph(%cond: tensor<i1> {onnx.name = "cond"}) -> (tensor<5xf32> {onnx.name = "output"}) {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%then_out) : (tensor<5xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Constant"() {value = dense<[5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%else_out) : (tensor<5xf32>) -> ()
    }) : (tensor<i1>) -> tensor<5xf32>
    "onnx.Return"(%res) : (tensor<5xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
