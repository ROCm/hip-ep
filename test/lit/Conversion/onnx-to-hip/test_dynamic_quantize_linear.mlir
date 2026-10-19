// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    return %arg0 : tensor<1x1x2x2xf32>
  }

  // The quantized Inception-v4 activation: NHWC-to-NCHW f32, then a
  // per-tensor uint8 dynamic quantize.
  func.func @dql_inception(%x: tensor<1x3x299x299xf32>)
      -> (tensor<1x3x299x299xui8>, tensor<f32>, tensor<ui8>) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%x)
        : (tensor<1x3x299x299xf32>)
          -> (tensor<1x3x299x299xui8>, tensor<f32>, tensor<ui8>)
    return %y, %scale, %zp
        : tensor<1x3x299x299xui8>, tensor<f32>, tensor<ui8>
  }

  // CHECK-LABEL: func.func @dql_inception
  // CHECK: hip.reduce_min
  // CHECK: hip.reduce_max
  // CHECK: hip.quantize_linear
  // CHECK-SAME: axis = 0
  // CHECK-NOT: onnx.DynamicQuantizeLinear

  // A degenerate `tensor<1xf32>` scale is reshaped from the rank-0 math.
  func.func @dql_rank1_scale(%x: tensor<1x1x2x2xf32>)
      -> (tensor<1x1x2x2xui8>, tensor<1xf32>, tensor<1xui8>) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%x)
        : (tensor<1x1x2x2xf32>)
          -> (tensor<1x1x2x2xui8>, tensor<1xf32>, tensor<1xui8>)
    return %y, %scale, %zp
        : tensor<1x1x2x2xui8>, tensor<1xf32>, tensor<1xui8>
  }

  // CHECK-LABEL: func.func @dql_rank1_scale
  // CHECK: hip.quantize_linear
  // CHECK: tensor.reshape
  // CHECK-NOT: onnx.DynamicQuantizeLinear

  // Non-f32 input is not this op. Leave it for a later pattern.
  func.func @dql_f16_rejected(%x: tensor<1x1x2x2xf16>)
      -> (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%x)
        : (tensor<1x1x2x2xf16>)
          -> (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>)
    return %y, %scale, %zp
        : tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>
  }

  // CHECK-LABEL: func.func @dql_f16_rejected
  // CHECK: onnx.DynamicQuantizeLinear

  // Rank 1 makes axis 1 out of range. Axis 0 stays legal.
  func.func @dql_rank1(%x: tensor<4xf32>)
      -> (tensor<4xui8>, tensor<f32>, tensor<ui8>) {
    %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%x)
        : (tensor<4xf32>) -> (tensor<4xui8>, tensor<f32>, tensor<ui8>)
    return %y, %scale, %zp : tensor<4xui8>, tensor<f32>, tensor<ui8>
  }

  // CHECK-LABEL: func.func @dql_rank1
  // CHECK: hip.quantize_linear
  // CHECK-SAME: axis = 0
  // CHECK-NOT: onnx.DynamicQuantizeLinear
}
