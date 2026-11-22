// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.QLinearConv -> hip.qlinear_conv
//
// Covers stride-2 3x3, depthwise 3x3, and 1x1 pointwise windows, plus
// per-output-channel weight scales. VALID, SAME_UPPER, and SAME_LOWER are
// rewritten to explicit pads. Per-channel input scales stay onnx.QLinearConv.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @qlinear_conv_stride2
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: group = 1 : i64
  // CHECK-SAME: kernel_shape = [3, 3]
  // CHECK-SAME: pads = [1, 1, 1, 1]
  // CHECK-SAME: strides = [2, 2]
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_stride2(
      %x: tensor<1x3x8x8xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x3x3x3xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x4x4xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]
    } : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>, tensor<4x3x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x4x4xui8>
    return %y : tensor<1x4x4x4xui8>
  }

  // CHECK-LABEL: func.func @qlinear_conv_depthwise
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: group = 4 : i64
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_depthwise(
      %x: tensor<1x4x6x6xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x1x3x3xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x6x6xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 4 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]
    } : (tensor<1x4x6x6xui8>, tensor<f32>, tensor<ui8>, tensor<4x1x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x6x6xui8>
    return %y : tensor<1x4x6x6xui8>
  }

  // CHECK-LABEL: func.func @qlinear_conv_pointwise
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: kernel_shape = [1, 1]
  // CHECK-NOT: tensor<6xi32>
  func.func @qlinear_conv_pointwise(
      %x: tensor<1x4x5x5xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<6x4x1x1xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>) -> tensor<1x6x5x5xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]
    } : (tensor<1x4x5x5xui8>, tensor<f32>, tensor<ui8>, tensor<6x4x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>) -> tensor<1x6x5x5xui8>
    return %y : tensor<1x6x5x5xui8>
  }

  // CHECK-LABEL: func.func @qlinear_conv_per_channel_weight
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: tensor<4xf32>
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_per_channel_weight(
      %x: tensor<1x2x3x3xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x2x1x1xi8>, %ws: tensor<4xf32>, %wz: tensor<4xi8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x3x3xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]
    } : (tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<4x2x1x1xi8>,
         tensor<4xf32>, tensor<4xi8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x3x3xui8>
    return %y : tensor<1x4x3x3xui8>
  }

  // Input 8, kernel 3, stride 2, output 4: pad budget is 1. SAME_UPPER
  // puts it at the end and ignores the pads attribute.
  // CHECK-LABEL: func.func @qlinear_conv_same_upper
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: pads = [0, 0, 1, 1]
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_same_upper(
      %x: tensor<1x3x8x8xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x3x3x3xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x4x4xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]
    } : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>, tensor<4x3x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x4x4xui8>
    return %y : tensor<1x4x4x4xui8>
  }

  // CHECK-LABEL: func.func @qlinear_conv_same_lower
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: pads = [1, 1, 0, 0]
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_same_lower(
      %x: tensor<1x3x8x8xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x3x3x3xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x4x4xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "SAME_LOWER", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], strides = [2, 2]
    } : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>, tensor<4x3x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x4x4xui8>
    return %y : tensor<1x4x4x4xui8>
  }

  // CHECK-LABEL: func.func @qlinear_conv_valid
  // CHECK: hip.qlinear_conv
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK-NOT: onnx.QLinearConv
  func.func @qlinear_conv_valid(
      %x: tensor<1x3x8x8xui8>, %xs: tensor<f32>, %xz: tensor<ui8>,
      %w: tensor<4x3x3x3xi8>, %ws: tensor<f32>, %wz: tensor<i8>,
      %ys: tensor<f32>, %yz: tensor<ui8>, %b: tensor<4xi32>) -> tensor<1x4x3x3xui8> {
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "VALID", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [2, 2]
    } : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>, tensor<4x3x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<4xi32>)
      -> tensor<1x4x3x3xui8>
    return %y : tensor<1x4x3x3xui8>
  }
}
