// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Opset 7-8 onnx.Upsample stores scales as an attribute. It lowers through
// hip.resize the same way as schema 9: asymmetric coordinates, nearest floor.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Upsample
// CHECK: llvm.func @wrap_resize
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x56x56xf32> {onnx.name = "input"})
      -> (tensor<1x128x112x112xf32> {onnx.name = "output"}) {
    %0 = "onnx.Upsample"(%arg0) {mode = "nearest", onnx_node_name = "upsample_node",
        scales = [1.000000e+00 : f32, 1.000000e+00 : f32,
                  2.000000e+00 : f32, 2.000000e+00 : f32]}
        : (tensor<1x128x56x56xf32>) -> tensor<1x128x112x112xf32>
    "onnx.Return"(%0) : (tensor<1x128x112x112xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
