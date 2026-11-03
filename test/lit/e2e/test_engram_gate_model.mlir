// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_engram_gate
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(%arg0: tensor<?x?x4x2560xf16> {onnx.name = "key"}, %arg1: tensor<?x?x4x2560xf16> {onnx.name = "query"}, %arg2: tensor<?x?x2560xf16> {onnx.name = "value"}) -> (tensor<?x?x4x2560xf16> {onnx.name = "out"}, tensor<?x?x4x2560xf16> {onnx.name = "normed"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4x2560xf16>} : () -> tensor<4x2560xf16>
    %1:2 = "onnx.Custom"(%arg0, %arg1, %arg2, %0, %0, %0) {function_name = "EngramGate", domain_name = "com.microsoft", epsilon = 9.99999997E-7 : f32, onnx_node_name = "/model/layers.1/ple/EngramGate"} : (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>, tensor<?x?x2560xf16>, tensor<4x2560xf16>, tensor<4x2560xf16>, tensor<4x2560xf16>) -> (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>)
    "onnx.Return"(%1#0, %1#1) : (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
