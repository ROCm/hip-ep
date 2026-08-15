// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// REQUIRES: hip_static_plugins
//
// RUN: split-file %s %t
// RUN: mkdir -p %t/out
// RUN: hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/out' --mlir-print-ir-after=hip-externalize-constants %t/ordered.mlir 2>&1 | FileCheck %s --check-prefix=ORDER
// RUN: not hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/out' %t/late.mlir 2>&1 | FileCheck %s --check-prefix=LATE

// The sample AfterConvertOnnxToHip pass inserts its carrier at the beginning
// of @main_graph, before both ONNX-origin carriers in structural walk order.
// The compiler-owned sweep order still places imported first, synthesized
// second, and the plugin carrier last.
// ORDER: IR Dump After ExternalizeConstantsPass
// ORDER: module attributes
// ORDER-SAME: hipdnn.constant_offsets = array<i64: 0, 64, 128>
// ORDER-SAME: hipdnn.constant_sizes = array<i64: 2, 4, 2>
// ORDER-DAG: memref.global "private" @hip_ext_constant_imported_weight_0{{.*}}hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 2 : i64}
// ORDER-DAG: memref.global "private" @hip_ext_constant_1{{.*}}hip.external_data = {index = 1 : i64, offset = 64 : i64, size = 4 : i64}
// ORDER-DAG: memref.global "private" @hip_ext_constant_sample_plugin_weight_2{{.*}}hip.external_data = {index = 2 : i64, offset = 128 : i64, size = 2 : i64}

// LATE: error: hip.constant survived past hip-externalize-constants

//--- ordered.mlir
module {
  func.func @main_graph(%input: tensor<2xf32> {onnx.name = "input"})
      -> (tensor<2xf32> {onnx.name = "output"},
          tensor<2xi8> {onnx.name = "weight"})
      attributes {hip_ep_sample.emit_constant} {
    %weight = "onnx.Constant"() {
      onnx_node_name = "imported_weight",
      value = dense<[11, 12]> : tensor<2xi8>
    } : () -> tensor<2xi8>
    %relu = "onnx.Relu"(%input) : (tensor<2xf32>) -> tensor<2xf32>
    "onnx.Return"(%relu, %weight)
        : (tensor<2xf32>, tensor<2xi8>) -> ()
  }
}

//--- late.mlir
module {
  func.func @main_graph(%input: tensor<2xf32> {onnx.name = "input"})
      -> tensor<2xf32> attributes {hip_ep_sample.emit_late_constant} {
    "onnx.Return"(%input) : (tensor<2xf32>) -> ()
  }
}
