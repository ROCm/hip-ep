// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.Sum -> onnx.Add -> hip.add -> wrap_elementwise.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Sum
module {
  func.func @main_graph(%arg0: tensor<1x4x2x2xf32> {onnx.name = "input"})
      -> (tensor<1x4x2x2xf32> {onnx.name = "output"}) {
    %bias = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1x4x1x1xf32>}
        : () -> tensor<1x4x1x1xf32>
    %0 = "onnx.Sum"(%arg0, %bias) {onnx_node_name = "sum_node"}
        : (tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x2x2xf32>
    "onnx.Return"(%0) : (tensor<1x4x2x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
