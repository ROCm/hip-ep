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
// 6. conv_dynamic_spatial — dynamic batch + dynamic spatial output dim
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
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
    return %arg0 : tensor<1x3x224x224xf32>
  }

  // --------------------------------------------------------------------------
  // 1. Basic conv with bias
  // --------------------------------------------------------------------------
  func.func @conv_basic(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x7xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %output : tensor<1x64x112x112xf32>
  }

  // CHECK-LABEL: func.func @conv_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x3x224x224xf32>, %[[W:.*]]: tensor<64x3x7x7xf32>, %[[B:.*]]: tensor<64xf32>) -> tensor<1x64x112x112xf32>
  // CHECK: tensor.empty() : tensor<1x64x112x112xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) outs({{.*}} : tensor<1x64x112x112xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}
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
      group = 2 : i64
    } : (tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) -> tensor<1x128x56x56xf32>
    return %output : tensor<1x128x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_grouped
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x32x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x56x56xf32>
  // CHECK: tensor.empty() : tensor<1x128x56x56xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x64x56x56xf32>, tensor<128x32x3x3xf32>, tensor<128xf32>) outs({{.*}} : tensor<1x128x56x56xf32>) {dilations = [1, 1], group = 2 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
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
      group = 64 : i64
    } : (tensor<1x64x56x56xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    return %output : tensor<1x64x56x56xf32>
  }

  // CHECK-LABEL: func.func @conv_depthwise
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) ins({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 64 : i64, kernel_shape = [3, 3]
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
      group = 1 : i64
    } : (tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    return %output : tensor<1x128x28x28xf32>
  }

  // CHECK-LABEL: func.func @conv_stride2
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>, %[[W:.*]]: tensor<128x64x3x3xf32>, %[[B:.*]]: tensor<128xf32>) -> tensor<1x128x28x28xf32>
  // CHECK: tensor.empty() : tensor<1x128x28x28xf32>
  // CHECK: hip.conv(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x64x56x56xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) outs({{.*}} : tensor<1x128x28x28xf32>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 5. Asymmetric stride [2, 3]
  // --------------------------------------------------------------------------
  func.func @conv_asymmetric_stride(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x3xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x75xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 3],
      strides = [2, 3],
      pads = [3, 1, 3, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x3xf32>, tensor<64xf32>) -> tensor<1x64x112x75xf32>
    return %output : tensor<1x64x112x75xf32>
  }

  // CHECK-LABEL: func.func @conv_asymmetric_stride
  // CHECK-SAME: !hip.context
  // CHECK: hip.conv({{.*}}) ins({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 3], pads = [3, 1, 3, 1], strides = [2, 3]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 6. Dynamic batch + dynamic spatial output dim (down-sampling front-end).
  //    Output H is sized at runtime from the conv formula:
  //      H' = (H + pad_begin + pad_end - dilation*(kernel-1) - 1)/stride + 1
  //    The static W dim (64) stays in the result type; only N and H are dynamic.
  // --------------------------------------------------------------------------
  func.func @conv_dynamic_spatial(%input: tensor<?x1x?x128xf16>, %weights: tensor<128x1x3x3xf16>, %bias: tensor<128xf16>) -> tensor<?x128x?x64xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3, 3],
      strides = [2, 2],
      pads = [1, 1, 1, 1],
      dilations = [1, 1],
      group = 1 : i64
    } : (tensor<?x1x?x128xf16>, tensor<128x1x3x3xf16>, tensor<128xf16>) -> tensor<?x128x?x64xf16>
    return %output : tensor<?x128x?x64xf16>
  }

  // CHECK-LABEL: func.func @conv_dynamic_spatial
  // CHECK-SAME: !hip.context
  // Spatial output dim resolved from tensor.dim of the input + arith, NOT the
  // conv result (would be a use-before-def).
  // CHECK: arith.floordivsi
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x128x?x64xf16>
  // CHECK: hip.conv({{.*}}) outs({{.*}} : tensor<?x128x?x64xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 7. Omitted kernel_shape is derived from static weight spatial dimensions.
  // --------------------------------------------------------------------------
  func.func @conv_derived_kernel(%input: tensor<1x3x8x8xf32>,
                                 %weights: tensor<4x3x3x5xf32>)
      -> tensor<1x4x6x4xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %output = "onnx.Conv"(%input, %weights, %none) {
      strides = [1, 1], pads = [0, 0, 0, 0]
    } : (tensor<1x3x8x8xf32>, tensor<4x3x3x5xf32>, none)
      -> tensor<1x4x6x4xf32>
    return %output : tensor<1x4x6x4xf32>
  }

  // CHECK-LABEL: func.func @conv_derived_kernel
  // CHECK: hip.conv
  // CHECK-SAME: kernel_shape = [3, 5]
}
