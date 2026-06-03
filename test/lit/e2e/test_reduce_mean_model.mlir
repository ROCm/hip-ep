// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_reduce_mean
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ReduceMean
module {
  func.func @main_graph(%arg0: tensor<8x128x512xf32> {onnx.name = "data"},
                        %arg1: tensor<1xi64> {onnx.name = "axes"}) -> (tensor<8x128xf32> {onnx.name = "reduced"}) {
    %0 = "onnx.ReduceMean"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64, onnx_node_name = "reduce_mean_node"} : (tensor<8x128x512xf32>, tensor<1xi64>) -> tensor<8x128xf32>
    "onnx.Return"(%0) : (tensor<8x128xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
