// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv operations with grouped convolutions are correctly lowered
// to hip.conv operations.
//
// This test validates:
// - Grouped convolution lowering (group > 1)
// - Depthwise convolution (group = num_channels)
// - Proper channel partitioning with groups
// - Weight tensor size validation for grouped convolutions
//
// Test cases:
// 1. Grouped conv (group=2): 64→128 channels, each group processes 32→64
// 2. Depthwise conv (group=64): Each of 64 channels processed independently
//
// Note: Grouped convolutions are common in MobileNet and ResNeXt architectures
// Expected: hip.conv operations with correct group attribute
// ============================================================================

// RUN: morphizen-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @conv_grouped_test(%input: tensor<1x64x56x56xf32>, %weights: tensor<128x32x3x3xf32>, %bias: tensor<128xf32>) -> tensor<1x128x56x56xf32> {
    // CHECK-LABEL: func.func @conv_grouped_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: memref<1x64x56x56xf32, 1>, %[[WEIGHTS:.*]]: memref<128x32x3x3xf32, 1>, %[[BIAS:.*]]: memref<128xf32, 1>, %[[OUTPUT_ARG:.*]]: memref<1x128x56x56xf32, 1>) -> i32

    // Grouped convolution with group=2
    // Input channels: 64, Output channels: 128, Groups: 2
    // Each group processes 32 input channels → 64 output channels
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 2 : si64
    } : (tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) -> tensor<1x128x56x56xf32>

    // CHECK: hip.conv(%[[CTX]], %[[INPUT]], %[[WEIGHTS]], %[[BIAS]], %[[OUTPUT_ARG]])
    // CHECK-SAME: {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
    // CHECK: arith.constant 0 : i32

    return %output : tensor<1x128x56x56xf32>
  }

  func.func @depthwise_conv(%input: tensor<1x64x56x56xf32>, %weights: tensor<64x1x3x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    // CHECK-LABEL: func.func @depthwise_conv
    // CHECK-SAME: -> i32

    // Depthwise convolution (group = num_channels)
    // Each of the 64 channels is processed independently with its own 3x3 filter
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 64 : si64
    } : (tensor<1x64x56x56xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>

    // CHECK: hip.conv(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {dilations = [1, 1], group = 64 : i64
    // CHECK: arith.constant 0 : i32

    return %output : tensor<1x64x56x56xf32>
  }
}
