// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip=conv-integer-quant-conv | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32> {
    return %arg0 : tensor<1x1x2x2xf32>
  }

  // Static int8 ConvInteger becomes one hip.quant_conv. No zero point, so
  // there is no correction convolution and no cast.
  func.func @conv_integer_quant_conv(
      %x: tensor<1x1x2x2xi8>, %w: tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xi8>, tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32>
    return %y : tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_quant_conv
  // CHECK-NOT: hip.cast
  // CHECK: hip.quant_conv
  // CHECK-SAME: tensor<1x1x2x2xi8>
  // CHECK-SAME: tensor<1x1x1x1xi8>
  // CHECK-SAME: tensor<1x1x2x2xi32>
  // CHECK-NOT: hip.quant_conv

  // uint8 is rebased by 128 into signless i8 before hip.quant_conv.
  func.func @conv_integer_quant_conv_ui8(
      %x: tensor<1x1x2x2xui8>, %w: tensor<1x1x1x1xui8>) -> tensor<1x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xui8>, tensor<1x1x1x1xui8>) -> tensor<1x1x2x2xi32>
    return %y : tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_quant_conv_ui8
  // CHECK: hip.cast
  // CHECK: hip.sub
  // CHECK: hip.quant_conv
  // CHECK-SAME: tensor<1x1x2x2xi8>
  // CHECK-SAME: tensor<1x1x1x1xi8>

  // A non-constant zero point needs the correction convolution:
  // conv(x, w) - conv(x_zp, w).
  func.func @conv_integer_quant_conv_zp(
      %x: tensor<1x1x2x2xi8>, %w: tensor<1x1x1x1xi8>, %xzp: tensor<i8>)
      -> tensor<1x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w, %xzp) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xi8>, tensor<1x1x1x1xi8>, tensor<i8>)
        -> tensor<1x1x2x2xi32>
    return %y : tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_quant_conv_zp
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK: hip.sub

  // A dynamic batch cannot be outlined, so it stays on hip.conv.
  func.func @conv_integer_quant_conv_dynamic(
      %x: tensor<?x1x2x2xi8>, %w: tensor<1x1x1x1xi8>) -> tensor<?x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1],
      strides = [1, 1],
      dilations = [1, 1],
      group = 1 : i64,
      auto_pad = "NOTSET"
    } : (tensor<?x1x2x2xi8>, tensor<1x1x1x1xi8>) -> tensor<?x1x2x2xi32>
    return %y : tensor<?x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_quant_conv_dynamic
  // CHECK: hip.cast
  // CHECK: hip.conv
  // CHECK-NOT: quant_conv
}
