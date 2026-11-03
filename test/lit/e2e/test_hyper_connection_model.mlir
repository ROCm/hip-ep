// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-DAG: llvm.func @wrap_hyper_connection_pre_mix
// CHECK-DAG: llvm.func @wrap_hyper_connection_post_mix
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(%arg0: tensor<?x?x10240xf16> {onnx.name = "streams"}, %arg1: tensor<?x?x10240xf16> {onnx.name = "pre_mix"}, %arg2: tensor<?x?x2560xf16> {onnx.name = "block"}, %arg3: tensor<?x?x4xf16> {onnx.name = "post_mix"}) -> (tensor<?x?x2560xf16> {onnx.name = "mixed"}, tensor<?x?x10240xf16> {onnx.name = "updated"}) {
    %0 = "onnx.Custom"(%arg0, %arg1) {function_name = "HyperConnectionPreMix", domain_name = "com.microsoft", num_branches = 4 : si64, reduction_scale = 2.500000e-01 : f32, onnx_node_name = "/model/layers.0/attn_hyper_connection/mixed/Mean"} : (tensor<?x?x10240xf16>, tensor<?x?x10240xf16>) -> tensor<?x?x2560xf16>
    %1 = "onnx.Custom"(%arg0, %arg2, %arg3) {function_name = "HyperConnectionPostMix", domain_name = "com.microsoft", num_branches = 4 : si64, onnx_node_name = "/model/layers.0/attn_hyper_connection/injection"} : (tensor<?x?x10240xf16>, tensor<?x?x2560xf16>, tensor<?x?x4xf16>) -> tensor<?x?x10240xf16>
    "onnx.Return"(%0, %1) : (tensor<?x?x2560xf16>, tensor<?x?x10240xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
