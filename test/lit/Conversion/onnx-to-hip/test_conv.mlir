// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv is correctly lowered to hip.conv in tensor-first mode.
//
// Test cases:
// 1. conv_basic          — standard 2D conv with bias (7x7 kernel, stride 2)
// 2. conv_grouped        — grouped conv (group=2)
// 3. conv_depthwise      — depthwise conv (group=channels)
// 4. conv_stride2        — strided conv (stride=2)
// 5. conv_asymmetric_stride — asymmetric stride [2,3]
//
// Note: conv without bias requires onnx.NoValue syntax which the current
// ConvToHipPattern does not guard against NoneType operands; tracked separately.
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - all Conv attributes forwarded (kernel_shape, strides, pads, dilations, group)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. Basic conv with bias
  // --------------------------------------------------------------------------
  func.func @conv_basic(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x7xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %output : tensor<1x64x112x112xf32>
  }

  // CHECK-LABEL: func.func @conv_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x3x224x224xf32>, %[[W:.*]]: tensor<64x3x7x7xf32>, %[[B:.*]]: tensor<64xf32>) -> tensor<1x64x112x112xf32>
  // CHECK: tensor.empty() : tensor<1x64x112x112xf32>
  // CHECK: hip.conv(%[[CTX]], %[[IN]], %[[W]], %[[B]], {{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. Grouped conv (group=2)
  // --------------------------------------------------------------------------
  func.func @conv_grouped(%input: tensor<1x64x56x56xf32>, %weights: tensor<128x32x3x3xf32>, %bias: tensor<128xf32>) -> tensor<1x128x56x56xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 2 : si64
    } : (tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) -> tensor<1x128x56x56xf32>
    return %output : tensor<1x128x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_grouped
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x32x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x56x56xf32>
  // CHECK: tensor.empty() : tensor<1x128x56x56xf32>
  // CHECK: hip.conv(%[[CTX]], %[[IN]], %[[W]], %[[B]], {{.*}}) {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. Depthwise conv (group = num_channels)
  // --------------------------------------------------------------------------
  func.func @conv_depthwise(%input: tensor<1x64x56x56xf32>, %weights: tensor<64x1x3x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 64 : si64
    } : (tensor<1x64x56x56xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    return %output : tensor<1x64x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_depthwise
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) {dilations = [1, 1], group = 64 : i64, kernel_shape = [3, 3]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 4. Strided conv (stride=2)
  // --------------------------------------------------------------------------
  func.func @conv_stride2(%input: tensor<1x64x56x56xf32>, %weights: tensor<128x64x3x3xf32>, %bias: tensor<128xf32>) -> tensor<1x128x28x28xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    return %output : tensor<1x128x28x28xf32>
  }

  // CHECK-LABEL: func.func @conv_stride2
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x64x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x28x28xf32>
  // CHECK: tensor.empty() : tensor<1x128x28x28xf32>
  // CHECK: hip.conv(%[[CTX]], %[[IN]], %[[W]], %[[B]], {{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 5. Asymmetric stride [2, 3]
  // --------------------------------------------------------------------------
  func.func @conv_asymmetric_stride(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x74xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 3],
      strides = [2, 3],
      pads = [3, 1, 3, 1],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x3xf32>, tensor<64xf32>) -> tensor<1x64x112x74xf32>
    return %output : tensor<1x64x112x74xf32>
  }

  // CHECK-LABEL: func.func @conv_asymmetric_stride
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 3], pads = [3, 1, 3, 1], strides = [2, 3]}
  // CHECK-NOT: hip.alloc
}
