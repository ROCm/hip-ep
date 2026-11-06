// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// com.microsoft.NhwcConv -> hip.transpose + hip.conv + hip.transpose
//
// Activations are NHWC and weights are [M, kH, kW, C/group]. hip.conv is
// NCHW with weights [M, C/group, kH, kW]. SAME_* padding stays onnx.Custom.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  // VAE decoder conv_in: dynamic N/H/W, 4 -> 512, 3x3, pad 1.
  // CHECK-LABEL: func.func @nhwc_conv_3x3_dynamic
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 1, 2]
  // CHECK-SAME: tensor<?x4x?x?xf16>
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 3, 1, 2]
  // CHECK-SAME: tensor<512x4x3x3xf16>
  // CHECK: hip.conv
  // CHECK-SAME: kernel_shape = [3, 3]
  // CHECK-SAME: pads = [1, 1, 1, 1]
  // CHECK-SAME: tensor<?x512x?x?xf16>
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 2, 3, 1]
  // CHECK-SAME: tensor<?x?x?x512xf16>
  func.func @nhwc_conv_3x3_dynamic(%x: tensor<?x?x?x4xf16>,
                                  %w: tensor<512x3x3x4xf16>,
                                  %b: tensor<512xf16>) -> tensor<?x?x?x512xf16> {
    %y = "onnx.Custom"(%x, %w, %b)
        <{function_name = "NhwcConv"}>
        {auto_pad = "NOTSET",
         dilations = [1, 1],
         domain_name = "com.microsoft",
         group = 1 : si64,
         kernel_shape = [3, 3],
         pads = [1, 1, 1, 1],
         strides = [1, 1]}
        : (tensor<?x?x?x4xf16>, tensor<512x3x3x4xf16>, tensor<512xf16>)
        -> tensor<?x?x?x512xf16>
    return %y : tensor<?x?x?x512xf16>
  }

  // VAE decoder post_quant_conv: 1x1, pad 0, channels stay 4.
  // CHECK-LABEL: func.func @nhwc_conv_1x1
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.conv
  // CHECK-SAME: kernel_shape = [1, 1]
  // CHECK-SAME: pads = [0, 0, 0, 0]
  // CHECK: hip.transpose
  // CHECK-SAME: perm = [0, 2, 3, 1]
  func.func @nhwc_conv_1x1(%x: tensor<1x8x8x4xf16>,
                           %w: tensor<4x1x1x4xf16>,
                           %b: tensor<4xf16>) -> tensor<1x8x8x4xf16> {
    %y = "onnx.Custom"(%x, %w, %b)
        <{function_name = "NhwcConv"}>
        {auto_pad = "NOTSET",
         dilations = [1, 1],
         domain_name = "com.microsoft",
         group = 1 : si64,
         kernel_shape = [1, 1],
         pads = [0, 0, 0, 0],
         strides = [1, 1]}
        : (tensor<1x8x8x4xf16>, tensor<4x1x1x4xf16>, tensor<4xf16>)
        -> tensor<1x8x8x4xf16>
    return %y : tensor<1x8x8x4xf16>
  }

  // Grouped: C/group is the weight's last dimension, so OIHW is [M, C/group, kH, kW].
  // CHECK-LABEL: func.func @nhwc_conv_grouped
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.conv
  // CHECK-SAME: tensor<4x2x3x3xf32>
  // CHECK-SAME: group = 2
  func.func @nhwc_conv_grouped(%x: tensor<1x8x8x4xf32>,
                               %w: tensor<4x3x3x2xf32>,
                               %b: tensor<4xf32>) -> tensor<1x8x8x4xf32> {
    %y = "onnx.Custom"(%x, %w, %b)
        <{function_name = "NhwcConv"}>
        {domain_name = "com.microsoft",
         group = 2 : si64,
         kernel_shape = [3, 3],
         pads = [1, 1, 1, 1],
         strides = [1, 1],
         dilations = [1, 1]}
        : (tensor<1x8x8x4xf32>, tensor<4x3x3x2xf32>, tensor<4xf32>)
        -> tensor<1x8x8x4xf32>
    return %y : tensor<1x8x8x4xf32>
  }

  // A static initializer is permuted to OIHW while lowering. OHWI
  // [1, 2, 1, 2] values 1,2,3,4 become OIHW [1, 2, 2, 1] values 1,3,2,4.
  // CHECK-LABEL: func.func @nhwc_conv_const_weight
  // CHECK: hip.constant
  // CHECK-SAME: 1.000000e+00
  // CHECK-SAME: 3.000000e+00
  // CHECK-SAME: 2.000000e+00
  // CHECK-SAME: 4.000000e+00
  // CHECK-SAME: tensor<1x2x2x1xf32>
  // CHECK: hip.conv
  // CHECK-SAME: tensor<1x2x2x1xf32>
  // CHECK-NOT: tensor<1x2x1x2xf32>
  func.func @nhwc_conv_const_weight(%x: tensor<1x4x4x2xf32>) -> tensor<1x3x4x1xf32> {
    %w = "onnx.Constant"() {value = dense<[[[[1.0, 2.0]], [[3.0, 4.0]]]]> : tensor<1x2x1x2xf32>}
        : () -> tensor<1x2x1x2xf32>
    %y = "onnx.Custom"(%x, %w)
        <{function_name = "NhwcConv"}>
        {auto_pad = "NOTSET",
         dilations = [1, 1],
         domain_name = "com.microsoft",
         group = 1 : si64,
         kernel_shape = [2, 1],
         pads = [0, 0, 0, 0],
         strides = [1, 1]}
        : (tensor<1x4x4x2xf32>, tensor<1x2x1x2xf32>) -> tensor<1x3x4x1xf32>
    return %y : tensor<1x3x4x1xf32>
  }

  // CHECK-LABEL: func.func @nhwc_conv_same_pad
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.conv
  func.func @nhwc_conv_same_pad(%x: tensor<1x8x8x4xf32>,
                                %w: tensor<4x3x3x4xf32>) -> tensor<1x8x8x4xf32> {
    %y = "onnx.Custom"(%x, %w)
        <{function_name = "NhwcConv"}>
        {auto_pad = "SAME_UPPER",
         domain_name = "com.microsoft",
         kernel_shape = [3, 3]}
        : (tensor<1x8x8x4xf32>, tensor<4x3x3x4xf32>) -> tensor<1x8x8x4xf32>
    return %y : tensor<1x8x8x4xf32>
  }
}
