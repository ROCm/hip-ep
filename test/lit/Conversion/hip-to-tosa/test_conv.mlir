// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify hip.conv lowers from ONNX layouts to a valid tosa.conv2d:
// NCHW -> NHWC, OIHW -> OHWI, and NHWC -> NCHW around the result.
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @conv
// CHECK: %[[INPUT:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x4x7x9xf16>) -> tensor<1x7x9x4xf16>
// CHECK: %[[WEIGHT:.*]] = tosa.transpose %arg2 {perms = array<i32: 0, 2, 3, 1>} : (tensor<6x4x3x2xf16>) -> tensor<6x3x2x4xf16>
// CHECK: %[[CONV:.*]] = tosa.conv2d %[[INPUT]], %[[WEIGHT]], %arg3
// CHECK-SAME: acc_type = f32
// CHECK-SAME: dilation = array<i64: 1, 1>
// CHECK-SAME: pad = array<i64: 1, 1, 2, 1>
// CHECK-SAME: stride = array<i64: 1, 2>
// CHECK: tosa.transpose %[[CONV]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x7x6x6xf16>) -> tensor<1x6x7x6xf16>
// CHECK-NOT: hip.conv
func.func @conv(%ctx: !hip.context, %input: tensor<1x4x7x9xf16>,
                %weight: tensor<6x4x3x2xf16>, %bias: tensor<6xf16>,
                %init: tensor<1x6x7x6xf16>) -> tensor<1x6x7x6xf16>
    attributes {rock.kernel} {
  %result = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x4x7x9xf16>, tensor<6x4x3x2xf16>, tensor<6xf16>)
      outs(%init : tensor<1x6x7x6xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 2],
       pads = [1, 2, 1, 1], strides = [1, 2]} : tensor<1x6x7x6xf16>
  return %result : tensor<1x6x7x6xf16>
}

// An omitted HIP bias becomes a zero TOSA bias.
// CHECK-LABEL: func.func @conv_without_bias
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK: tosa.conv2d {{.*}}, %[[ZERO]]
// CHECK-NOT: hip.conv
func.func @conv_without_bias(
    %ctx: !hip.context, %input: tensor<1x3x5x5xf32>,
    %weight: tensor<4x3x3x3xf32>, %init: tensor<1x4x3x3xf32>)
    -> tensor<1x4x3x3xf32> attributes {rock.kernel} {
  %result = hip.conv(%ctx) ins(
      %input, %weight : tensor<1x3x5x5xf32>, tensor<4x3x3x3xf32>)
      outs(%init : tensor<1x4x3x3xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x4x3x3xf32>
  return %result : tensor<1x4x3x3xf32>
}

// ResNet stem: ONNX floors 112 = floor((224 - 7 + 6) / 2) + 1, but TOSA
// requires exact division. The trailing pad absorbs the odd remainder.
// CHECK-LABEL: func.func @conv_stride_pad_absorbs_remainder
// CHECK: tosa.conv2d
// CHECK-SAME: pad = array<i64: 3, 2, 3, 2>
// CHECK-SAME: stride = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x112x112x64xf16>
// CHECK-NOT: hip.conv
func.func @conv_stride_pad_absorbs_remainder(
    %ctx: !hip.context, %input: tensor<1x3x224x224xf16>,
    %weight: tensor<64x3x7x7xf16>, %bias: tensor<64xf16>,
    %init: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    attributes {rock.kernel} {
  %result = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x3x224x224xf16>, tensor<64x3x7x7xf16>, tensor<64xf16>)
      outs(%init : tensor<1x64x112x112xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7],
       pads = [3, 3, 3, 3], strides = [2, 2]} : tensor<1x64x112x112xf16>
  return %result : tensor<1x64x112x112xf16>
}

// ResNet downsample shortcut: a strided 1x1 kernel has no padding to give
// back, so the row and column the convolution never reads are sliced off.
// CHECK-LABEL: func.func @conv_stride_crops_input
// CHECK: %[[NHWC:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>}
// CHECK: %[[CROP:.*]] = tosa.slice %[[NHWC]], {{.*}} -> tensor<1x55x55x256xf16>
// CHECK: tosa.conv2d %[[CROP]]
// CHECK-SAME: pad = array<i64: 0, 0, 0, 0>
// CHECK-SAME: -> tensor<1x28x28x512xf16>
// CHECK-NOT: hip.conv
func.func @conv_stride_crops_input(
    %ctx: !hip.context, %input: tensor<1x256x56x56xf16>,
    %weight: tensor<512x256x1x1xf16>, %bias: tensor<512xf16>,
    %init: tensor<1x512x28x28xf16>) -> tensor<1x512x28x28xf16>
    attributes {rock.kernel} {
  %result = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x256x56x56xf16>, tensor<512x256x1x1xf16>, tensor<512xf16>)
      outs(%init : tensor<1x512x28x28xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [2, 2]} : tensor<1x512x28x28xf16>
  return %result : tensor<1x512x28x28xf16>
}

// -----

func.func @dynamic_conv(
    %ctx: !hip.context, %input: tensor<1x3x?x5xf16>,
    %weight: tensor<4x3x3x3xf16>, %bias: tensor<4xf16>,
    %init: tensor<1x4x?x3xf16>) -> tensor<1x4x?x3xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.conv'}}
  %result = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x3x?x5xf16>, tensor<4x3x3x3xf16>, tensor<4xf16>)
      outs(%init : tensor<1x4x?x3xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x4x?x3xf16>
  return %result : tensor<1x4x?x3xf16>
}

// -----

func.func @grouped_conv(
    %ctx: !hip.context, %input: tensor<1x4x7x9xf16>,
    %weight: tensor<6x2x3x2xf16>, %bias: tensor<6xf16>,
    %init: tensor<1x6x7x6xf16>) -> tensor<1x6x7x6xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.conv'}}
  %result = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x4x7x9xf16>, tensor<6x2x3x2xf16>, tensor<6xf16>)
      outs(%init : tensor<1x6x7x6xf16>)
      {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 2],
       pads = [1, 2, 1, 1], strides = [1, 2]} : tensor<1x6x7x6xf16>
  return %result : tensor<1x6x7x6xf16>
}

// -----

func.func @kernel_shape_mismatch(
    %ctx: !hip.context, %input: tensor<1x3x5x5xf32>,
    %weight: tensor<4x3x3x3xf32>, %init: tensor<1x4x3x3xf32>)
    -> tensor<1x4x3x3xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.conv'}}
  %result = hip.conv(%ctx) ins(
      %input, %weight : tensor<1x3x5x5xf32>, tensor<4x3x3x3xf32>)
      outs(%init : tensor<1x4x3x3xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x4x3x3xf32>
  return %result : tensor<1x4x3x3xf32>
}
