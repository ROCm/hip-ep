// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// End-to-end guard for an onnx.If whose outputs have dynamic sizes that
// differ per branch, as in merged decoders (`present.*` is past + new rows on
// the cache branch, new rows otherwise). The o_init sizes must be resolved to
// host arithmetic before bufferization, and the outputs must reach the
// output allocator; the hipdnn_ep_run_if call proves the GPU path compiled.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 2
// CHECK-NOT: hip.if_result_dim
// CHECK: llvm.func @hipdnn_ep_run_if
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.If
module {
  func.func @main_graph(%cond: tensor<i1> {onnx.name = "use_cache"},
                        %x: tensor<?x4xf32> {onnx.name = "x"},
                        %past: tensor<?x4xf32> {onnx.name = "past"})
      -> (tensor<?x4xf32> {onnx.name = "present"},
          tensor<?x4xf32> {onnx.name = "side"}) {
    %res:2 = "onnx.If"(%cond) ({
    ^bb0():
      %cat = "onnx.Concat"(%past, %x) {axis = 0 : si64}
          : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
      %dummy = "onnx.Constant"() {value = dense<0.0> : tensor<0x4xf32>}
          : () -> tensor<0x4xf32>
      "onnx.Yield"(%cat, %dummy) : (tensor<?x4xf32>, tensor<0x4xf32>) -> ()
    }, {
    ^bb0():
      %r = "onnx.Relu"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %n = "onnx.Neg"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      "onnx.Yield"(%r, %n) : (tensor<?x4xf32>, tensor<?x4xf32>) -> ()
    }) : (tensor<i1>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
    "onnx.Return"(%res#0, %res#1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
