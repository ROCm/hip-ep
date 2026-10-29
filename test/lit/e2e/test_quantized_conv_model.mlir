// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// DynamicQuantizeLinear followed by ConvInteger, the head of a quantized
// conv graph. Both ops decompose to HIP ops that bufferize; hip.conv and
// hip.quantize_linear then lower to their runtime wrappers.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.DynamicQuantizeLinear
// CHECK-NOT: onnx.ConvInteger
// CHECK-DAG: llvm.func @wrap_conv
// CHECK-DAG: llvm.func @wrap_quantize_linear
// CHECK-DAG: llvm.func @wrap_reduce_min
// CHECK-DAG: llvm.func @wrap_reduce_max
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(
      %arg0: tensor<1x1x2x2xf32> {onnx.name = "input"},
      %arg1: tensor<1x1x1x1xui8> {onnx.name = "weight"},
      %arg2: tensor<ui8> {onnx.name = "w_zp"})
      -> (tensor<1x1x2x2xi32> {onnx.name = "output"}) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%arg0) {
      onnx_node_name = "dynamic_quantize_linear"
    } : (tensor<1x1x2x2xf32>)
        -> (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>)
    %conv = "onnx.ConvInteger"(%y, %arg1, %zp, %arg2) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET",
      onnx_node_name = "conv_integer"
    } : (tensor<1x1x2x2xui8>, tensor<1x1x1x1xui8>, tensor<ui8>, tensor<ui8>)
        -> tensor<1x1x2x2xi32>
    "onnx.Return"(%conv) : (tensor<1x1x2x2xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
