// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Gated-MLP SwiGLU as a gated-decoder export emits it: three primitive
// elementwise nodes over the MLP intermediate width. Reproduced from
// Llama-3.1-8B (32 layers, intermediate 14336, fp16), whose export ships
// exactly 32 Sigmoid and 64 Mul nodes for these chains and nothing else.
// Batch and sequence are dynamic, matching that export. Conversion then
// sizes the outer mul with tensor.dim of the inner product, and
// hip-fusion-transform runs before canonicalize, so the fusion has to
// accept those shape queries itself.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip lowers Sigmoid and Mul one-to-one, then
//    hip-fusion-transform collapses that HIP chain into one hip.swiglu, so no
//    sigmoid or elementwise-multiply kernel is left.
// 2. canonicalize / memory-pooling: pool the output buffer into one allocation
// 3. convert-hip-to-llvm: hip.swiglu -> wrap_swiglu
// 4. generate-interface: create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_swiglu
// CHECK-DAG: llvm.func @inference_init
// CHECK-DAG: llvm.func @inference_compute
// CHECK-DAG: llvm.func @inference_cleanup
// CHECK-DAG: llvm.func @inference_get_metadata_json
// CHECK-NOT: llvm.func @wrap_elementwise
module {
  func.func @main_graph(%gate: tensor<?x?x14336xf16> {onnx.name = "gate"},
                        %up: tensor<?x?x14336xf16> {onnx.name = "up"})
      -> (tensor<?x?x14336xf16> {onnx.name = "y"}) {
    %s = "onnx.Sigmoid"(%gate) {onnx_node_name = "act_fn.Sigmoid"}
        : (tensor<?x?x14336xf16>) -> tensor<?x?x14336xf16>
    %a = "onnx.Mul"(%gate, %s) {onnx_node_name = "act_fn.Mul"}
        : (tensor<?x?x14336xf16>, tensor<?x?x14336xf16>) -> tensor<?x?x14336xf16>
    %y = "onnx.Mul"(%a, %up) {onnx_node_name = "mlp.Mul"}
        : (tensor<?x?x14336xf16>, tensor<?x?x14336xf16>) -> tensor<?x?x14336xf16>
    "onnx.Return"(%y) : (tensor<?x?x14336xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
