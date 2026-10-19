// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    return %arg0 : tensor<1x1x2x2xf32>
  }

  // First conv of quantized Inception-v4: ui8 1x3x299x299, 32x3x3x3,
  // stride 2, scalar zero points, i32 result 1x32x149x149.
  func.func @conv_integer_inception(
      %x: tensor<1x3x299x299xui8>, %w: tensor<32x3x3x3xui8>,
      %xzp: tensor<ui8>, %wzp: tensor<ui8>) -> tensor<1x32x149x149xi32> {
    %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x3x299x299xui8>, tensor<32x3x3x3xui8>, tensor<ui8>, tensor<ui8>)
        -> tensor<1x32x149x149xi32>
    return %y : tensor<1x32x149x149xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_inception
  // CHECK: hip.cast
  // CHECK: hip.sub
  // CHECK: hip.conv
  // CHECK-SAME: tensor<1x3x299x299xf32>
  // CHECK-SAME: tensor<32x3x3x3xf32>
  // CHECK: hip.cast
  // CHECK-SAME: tensor<1x32x149x149xi32>
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
