// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Gated-MLP SwiGLU as a gated-decoder export emits it: MatMul projections
// of one hidden state, then Sigmoid and two Muls over the MLP intermediate.
// Batch and sequence are dynamic. The projections are in the graph so those
// dynamic axes share the hidden state; two unbound inputs of the same type
// would not. Conversion sizes the outer mul with tensor.dim of the inner
// product, and hip-fusion-transform runs before canonicalize, so the fusion
// has to accept those shape queries itself.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip lowers MatMul, Sigmoid, and Mul, then
//    hip-fusion-transform collapses the activation chain into one hip.swiglu,
//    so no sigmoid or elementwise-multiply kernel is left.
// 2. canonicalize / memory-pooling: pool the output buffer into one allocation
// 3. convert-hip-to-llvm: hip.swiglu -> wrap_swiglu
// 4. generate-interface: create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_swiglu
// CHECK-DAG: llvm.func @inference_init
// CHECK-DAG: llvm.func @inference_compute
// CHECK-DAG: llvm.func @inference_cleanup
// CHECK-DAG: llvm.func @inference_get_metadata_json
// CHECK-NOT: llvm.func @wrap_elementwise
module {
  func.func @main_graph(%hidden: tensor<?x?x4096xf16> {onnx.name = "hidden"},
                        %w_gate: tensor<4096x14336xf16> {onnx.name = "w_gate"},
                        %w_up: tensor<4096x14336xf16> {onnx.name = "w_up"})
      -> (tensor<?x?x14336xf16> {onnx.name = "y"}) {
    %gate = "onnx.MatMul"(%hidden, %w_gate) {onnx_node_name = "gate_proj.MatMul"}
        : (tensor<?x?x4096xf16>, tensor<4096x14336xf16>) -> tensor<?x?x14336xf16>
    %up = "onnx.MatMul"(%hidden, %w_up) {onnx_node_name = "up_proj.MatMul"}
        : (tensor<?x?x4096xf16>, tensor<4096x14336xf16>) -> tensor<?x?x14336xf16>
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
