// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// A runtime i32 Slice graph output is allocated from its exact normalized
// extents before wrap_slice executes. One grouped readback covers starts+ends.
// CHECK-COUNT-1: llvm.call @hipdnn_ep_readback_control
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: llvm.call @wrap_slice
// CHECK-NOT: llvm.call @hipdnn_ep_readback_control
module {
  func.func @main_graph(
      %data: tensor<1x8x4xf32> {onnx.name = "data"},
      %starts: tensor<1xi32> {onnx.name = "starts"},
      %ends: tensor<1xi32> {onnx.name = "ends"})
      -> (tensor<?x?x?xf32> {onnx.name = "output"}) {
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %result = "onnx.Slice"(%data, %starts, %ends, %axes, %steps)
        : (tensor<1x8x4xf32>, tensor<1xi32>, tensor<1xi32>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<?x?x?xf32>
    "onnx.Return"(%result) : (tensor<?x?x?xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
