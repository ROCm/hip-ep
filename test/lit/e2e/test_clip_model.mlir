// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_clip
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Clip
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "x"},
                        %arg1: tensor<f32> {onnx.name = "lo"},
                        %arg2: tensor<f32> {onnx.name = "hi"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
    %0 = "onnx.Clip"(%arg0, %arg1, %arg2) {onnx_node_name = "clip_node"} : (tensor<3x4xf32>, tensor<f32>, tensor<f32>) -> tensor<3x4xf32>
    "onnx.Return"(%0) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
