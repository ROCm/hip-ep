// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// EmbedLayerNormalization must bufferize. The composed gather, add, and
// layer-norm ops lower through the production pipeline.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK-NOT: onnx.Custom
// CHECK-DAG: llvm.func @wrap_gather
// CHECK-DAG: llvm.func @wrap_layer_normalization
// CHECK-DAG: llvm.func @wrap_reduce_sum
// CHECK: llvm.func @inference_compute
module {
  func.func @main_graph(
      %ids: tensor<1x4xi32> {onnx.name = "input_ids"},
      %segments: tensor<1x4xi32> {onnx.name = "segment_ids"},
      %word: tensor<16x8xf16> {onnx.name = "word_embedding"},
      %position: tensor<8x8xf16> {onnx.name = "position_embedding"},
      %segment: tensor<2x8xf16> {onnx.name = "segment_embedding"},
      %gamma: tensor<8xf16> {onnx.name = "gamma"},
      %beta: tensor<8xf16> {onnx.name = "beta"},
      %mask: tensor<1x4xi32> {onnx.name = "mask"},
      %position_ids: tensor<1x4xi32> {onnx.name = "position_ids"})
      -> (tensor<1x4x8xf16> {onnx.name = "output"},
          tensor<1xi32> {onnx.name = "mask_index"}) {
    %0:2 = "onnx.Custom"(%ids, %segments, %word, %position, %segment,
                          %gamma, %beta, %mask, %position_ids) {
      function_name = "EmbedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.000000e-12 : f32
    } : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<16x8xf16>, tensor<8x8xf16>,
         tensor<2x8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<1x4xi32>,
         tensor<1x4xi32>) -> (tensor<1x4x8xf16>, tensor<1xi32>)
    "onnx.Return"(%0#0, %0#1) : (tensor<1x4x8xf16>, tensor<1xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
