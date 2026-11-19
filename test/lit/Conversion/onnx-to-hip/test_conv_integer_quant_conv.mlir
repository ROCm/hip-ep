// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip=conv-integer-quant-conv | FileCheck %s

// hip.quant_conv is outlined only from @main_graph's entry block. The same
// static ConvInteger in any other function stays on hip.conv.
module {
  // A uint8 1x1 convolution of x=2, w=3 with both zero points omitted is
  // (2-0)*(3-0)=6. Rebasing each operand by 128 gives (-126)*(-125)=15750
  // unless the omitted zero points are materialized as 0, which rebase to
  // -128 and restore the product through the correction convolutions.
  func.func @main_graph(
      %x: tensor<1x1x2x2xi8>, %w: tensor<1x1x1x1xi8>,
      %xu: tensor<1x1x2x2xui8>, %wu: tensor<1x1x1x1xui8>,
      %xd: tensor<?x1x2x2xi8>, %xzp: tensor<i8>,
      %x1: tensor<1x1x4xi8>, %w1: tensor<2x1x2xi8>, %wz1: tensor<2x1x1xi8>,
      %xm: tensor<1x1x2x2xui8>, %wm: tensor<1x1x1x1xi8>) -> (
        tensor<1x1x2x2xi32>, tensor<1x1x2x2xi32>, tensor<?x1x2x2xi32>,
        tensor<1x1x2x2xi32>, tensor<1x2x3xi32>, tensor<1x1x2x2xi32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %signed = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xi8>, tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32>
    %omitted = "onnx.ConvInteger"(%xu, %wu) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xui8>, tensor<1x1x1x1xui8>) -> tensor<1x1x2x2xi32>
    %dynamic = "onnx.ConvInteger"(%xd, %w) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<?x1x2x2xi8>, tensor<1x1x1x1xi8>) -> tensor<?x1x2x2xi32>
    %with_zp = "onnx.ConvInteger"(%x, %w, %xzp) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xi8>, tensor<1x1x1x1xi8>, tensor<i8>)
        -> tensor<1x1x2x2xi32>
    %rank3_zp = "onnx.ConvInteger"(%x1, %w1, %none, %wz1) {
      kernel_shape = [2], strides = [1], dilations = [1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x4xi8>, tensor<2x1x2xi8>, none, tensor<2x1x1xi8>)
        -> tensor<1x2x3xi32>
    %mixed = "onnx.ConvInteger"(%xm, %wm) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xui8>, tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32>
    return %signed, %omitted, %dynamic, %with_zp, %rank3_zp, %mixed :
        tensor<1x1x2x2xi32>, tensor<1x1x2x2xi32>, tensor<?x1x2x2xi32>,
        tensor<1x1x2x2xi32>, tensor<1x2x3xi32>, tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @main_graph
  // Signed, no zero point: one convolution and no rebase cast.
  // CHECK-NOT: hip.cast
  // CHECK: hip.quant_conv
  // CHECK-SAME: tensor<1x1x2x2xi8>
  // CHECK-SAME: tensor<1x1x1x1xi8>
  // CHECK-NOT: hip.quant_conv

  // Both uint8 zero points omitted: materialized 0, rebased by 128, and one
  // correction convolution per zero point plus the product of the two.
  // CHECK: hip.constant {{.*}}dense<0>
  // CHECK: hip.constant {{.*}}dense<128>
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK-NOT: hip.quant_conv

  // A dynamic batch cannot be outlined.
  // CHECK: = hip.conv
  // CHECK: tensor<?x1x2x2xf32>
  // CHECK-NOT: quant_conv

  // A non-constant zero point needs conv(x, w) - conv(x_zp, w).
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK: hip.sub

  // The rank-3 weight zero point is expanded with the weight, not reshaped
  // down to a scalar.
  // CHECK: tensor.expand_shape {{.*}} : tensor<2x1x1xi8> into tensor<2x1x1x1xi8>
  // CHECK-NOT: tensor.reshape {{.*}}tensor<2x1x1xi8>

  // Only the unsigned operand of a mixed pair gets a materialized zero point,
  // so there are two convolutions and no third.
  // CHECK: hip.cast
  // CHECK: hip.quant_conv
  // CHECK: hip.quant_conv
  // CHECK: hip.sub
  // CHECK-NOT: hip.quant_conv

  func.func @conv_integer_not_in_main(
      %x: tensor<1x1x2x2xi8>, %w: tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32> {
    %y = "onnx.ConvInteger"(%x, %w) {
      kernel_shape = [1, 1], strides = [1, 1], dilations = [1, 1],
      group = 1 : i64, auto_pad = "NOTSET"
    } : (tensor<1x1x2x2xi8>, tensor<1x1x1x1xi8>) -> tensor<1x1x2x2xi32>
    return %y : tensor<1x1x2x2xi32>
  }

  // CHECK-LABEL: func.func @conv_integer_not_in_main
  // CHECK: = hip.conv
  // CHECK-NOT: quant_conv
}
