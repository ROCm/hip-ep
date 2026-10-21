// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    return %arg0 : tensor<1x1x2x2xf32>
  }

  // Rank-4 NCHW, stride 2, scalar zero points. The reduction is 27, which
  // still fits the f32 mantissa.
  func.func @conv_integer_stride2(
      %x: tensor<1x3x8x8xui8>, %w: tensor<4x3x3x3xui8>,
      %xzp: tensor<ui8>, %wzp: tensor<ui8>) -> tensor<1x4x3x3xi32> {
    %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x3x8x8xui8>, tensor<4x3x3x3xui8>, tensor<ui8>, tensor<ui8>)
        -> tensor<1x4x3x3xi32>
    return %y : tensor<1x4x3x3xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_stride2
  // CHECK: hip.cast
  // CHECK: hip.sub
  // CHECK: hip.conv
  // CHECK-SAME: tensor<1x3x8x8xf32>
  // CHECK-SAME: tensor<4x3x3x3xf32>
  // CHECK: hip.cast
  // CHECK-SAME: tensor<1x4x3x3xi32>
  // CHECK-NOT: onnx.ConvInteger

  // Per-output-channel weight zero point is reshaped to [M, 1, 1, 1] so it
  // broadcasts on C, not on W.
  func.func @conv_integer_per_channel(
      %x: tensor<1x1x2x2xui8>, %w: tensor<2x1x1x1xui8>,
      %xzp: tensor<ui8>, %wzp: tensor<2xui8>) -> tensor<1x2x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x1x2x2xui8>, tensor<2x1x1x1xui8>, tensor<ui8>, tensor<2xui8>)
        -> tensor<1x2x2x2xi32>
    return %y : tensor<1x2x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_per_channel
  // CHECK: tensor<2x1x1x1xf32>
  // CHECK: hip.conv
  // CHECK-NOT: onnx.ConvInteger

  // Omitted zero points are defined as 0, so the subtract is dropped.
  func.func @conv_integer_no_zp(
      %x: tensor<1x1x4x4xi8>, %w: tensor<1x1x3x3xi8>) -> tensor<1x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      group = 1 : i64
    } : (tensor<1x1x4x4xi8>, tensor<1x1x3x3xi8>) -> tensor<1x1x2x2xi32>
    return %y : tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_no_zp
  // CHECK-NOT: hip.sub
  // CHECK: hip.conv
  // CHECK-SAME: tensor<1x1x4x4xf32>
  // CHECK-NOT: onnx.ConvInteger

  // Rank-3 NCL ConvInteger uses the shared rank-4 kernel path. Both the
  // accumulation type and output are widened to NC1L, then collapsed to NCL.
  func.func @conv_integer_1d(
      %x: tensor<1x80x3000xui8>, %w: tensor<1280x80x3xi8>,
      %xzp: tensor<ui8>, %wzp: tensor<i8>) -> tensor<1x1280x3000xi32> {
    %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp) {
      kernel_shape = [3],
      strides = [1],
      pads = [1, 1],
      dilations = [1],
      group = 1 : i64
    } : (tensor<1x80x3000xui8>, tensor<1280x80x3xi8>, tensor<ui8>, tensor<i8>)
        -> tensor<1x1280x3000xi32>
    return %y : tensor<1x1280x3000xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_1d
  // CHECK: tensor.expand_shape
  // CHECK-SAME: tensor<1x80x3000xf32> into tensor<1x80x1x3000xf32>
  // CHECK: tensor.expand_shape
  // CHECK-SAME: tensor<1280x80x3xf32> into tensor<1280x80x1x3xf32>
  // CHECK: hip.conv
  // CHECK-SAME: kernel_shape = [1, 3]
  // CHECK-SAME: pads = [0, 1, 0, 1]
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: onnx.ConvInteger

  // Rank 2 has no 4D broadcast path for the zero-point subtract.
  func.func @conv_integer_rank2_rejected(
      %x: tensor<1x4xui8>, %w: tensor<2x4xui8>) -> tensor<1x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1],
      group = 1 : i64
    } : (tensor<1x4xui8>, tensor<2x4xui8>) -> tensor<1x2xi32>
    return %y : tensor<1x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_rank2_rejected
  // CHECK: onnx.ConvInteger

  // Cin = 259 is the first reduction whose uint8 dot product is not an exact f32.
  func.func @conv_integer_wide_k(
      %x: tensor<1x259x1x1xui8>, %w: tensor<1x259x1x1xui8>) -> tensor<1x1x1x1xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      pads = [0, 0, 0, 0],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x259x1x1xui8>, tensor<1x259x1x1xui8>) -> tensor<1x1x1x1xi32>
    return %y : tensor<1x1x1x1xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_wide_k
  // CHECK: hip.conv
  // CHECK-SAME: tensor<1x259x1x1xi32>
  // CHECK-SAME: tensor<1x1x1x1xi32>
  // CHECK-NOT: onnx.ConvInteger
}
