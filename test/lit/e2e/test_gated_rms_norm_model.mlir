// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_gated_rms_norm
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(%arg0: tensor<?x?x6144xf16> {onnx.name = "x"}, %arg1: tensor<?x?x6144xf16> {onnx.name = "z"}) -> (tensor<?x?x6144xf16> {onnx.name = "y"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<128xf16>} : () -> tensor<128xf16>
    %1 = "onnx.Custom"(%arg0, %0, %arg1) {function_name = "GatedRMSNorm", domain_name = "com.microsoft", activation = "sigmoid", epsilon = 9.99999997E-7 : f32, onnx_node_name = "/model/layers.0/linear_attn/GatedRMSNorm"} : (tensor<?x?x6144xf16>, tensor<128xf16>, tensor<?x?x6144xf16>) -> tensor<?x?x6144xf16>
    "onnx.Return"(%1) : (tensor<?x?x6144xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
