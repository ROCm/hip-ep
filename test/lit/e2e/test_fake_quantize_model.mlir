// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// A per-tensor QuantizeLinear read only by a DequantizeLinear with the same
// scale and zero point runs through the full pipeline as one
// wrap_fake_quantize call; the integer codes never get a buffer.

// CHECK: module attributes {
// CHECK-NOT: llvm.call @wrap_quantize_linear
// CHECK-NOT: llvm.call @wrap_dequantize_linear
// CHECK: llvm.call @wrap_fake_quantize
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.QuantizeLinear
// CHECK-NOT: onnx.DequantizeLinear

module {
  func.func @main_graph(%x: tensor<1x64x32xf32> {onnx.name = "x"})
      -> (tensor<1x64x32xf32> {onnx.name = "output"}) {
    %scale = "onnx.Constant"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      saturate = 1 : si64,
      onnx_node_name = "QuantizeLinear_0"
    } : (tensor<1x64x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x64x32xi8>
    %dq = "onnx.DequantizeLinear"(%q, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "DequantizeLinear_0"
    } : (tensor<1x64x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x64x32xf32>
    "onnx.Return"(%dq) : (tensor<1x64x32xf32>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
