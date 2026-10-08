// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.PRelu -> hip.prelu -> wrap_prelu. Per-channel slope broadcasts onto X.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.PRelu
// CHECK: llvm.func @wrap_prelu
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x2x4x4xf32> {onnx.name = "input"})
      -> (tensor<1x2x4x4xf32> {onnx.name = "output"}) {
    %slope = "onnx.Constant"() {value = dense<2.500000e-01> : tensor<1x2x1x1xf32>}
        : () -> tensor<1x2x1x1xf32>
    %0 = "onnx.PRelu"(%arg0, %slope) {onnx_node_name = "prelu_node"}
        : (tensor<1x2x4x4xf32>, tensor<1x2x1x1xf32>) -> tensor<1x2x4x4xf32>
    "onnx.Return"(%0) : (tensor<1x2x4x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
