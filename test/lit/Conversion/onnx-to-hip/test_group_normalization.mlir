// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify ONNX GroupNormalization (schema 21, per-channel scale and bias)
// lowers to hip.group_norm with channels_last = 0 and no activation.
//
// 1. Static NCHW f16
// 2. Dynamic batch and spatial extents, f32
// 3. Schema-18 per-group scale/bias (length G != C) stays unconverted
// 4. num_groups == C: both schema readings agree, so it converts

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @group_normalization_static(%X: tensor<2x8x4x4xf16>,
                                        %scale: tensor<8xf16>,
                                        %bias: tensor<8xf16>) -> tensor<2x8x4x4xf16> {
    %Y = "onnx.GroupNormalization"(%X, %scale, %bias)
        {num_groups = 4 : si64, epsilon = 9.99999974E-6 : f32}
        : (tensor<2x8x4x4xf16>, tensor<8xf16>, tensor<8xf16>) -> tensor<2x8x4x4xf16>
    return %Y : tensor<2x8x4x4xf16>
  }

  // CHECK-LABEL: func.func @group_normalization_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x8x4x4xf16>, %[[S:.*]]: tensor<8xf16>, %[[B:.*]]: tensor<8xf16>)
  // CHECK-NOT: onnx.GroupNormalization
  // CHECK: hip.group_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[S]], %[[B]] :
  // CHECK-SAME: channels_last = 0
  // CHECK-SAME: groups = 4

  func.func @group_normalization_dynamic(%X: tensor<?x32x?x?xf32>,
                                         %scale: tensor<32xf32>,
                                         %bias: tensor<32xf32>) -> tensor<?x32x?x?xf32> {
    %Y = "onnx.GroupNormalization"(%X, %scale, %bias) {num_groups = 8 : si64}
        : (tensor<?x32x?x?xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<?x32x?x?xf32>
    return %Y : tensor<?x32x?x?xf32>
  }

  // CHECK-LABEL: func.func @group_normalization_dynamic
  // CHECK-NOT: onnx.GroupNormalization
  // CHECK: hip.group_norm
  // CHECK-SAME: channels_last = 0
  // CHECK-SAME: groups = 8

  func.func @group_normalization_per_group(%X: tensor<1x8x4x4xf32>,
                                           %scale: tensor<4xf32>,
                                           %bias: tensor<4xf32>) -> tensor<1x8x4x4xf32> {
    %Y = "onnx.GroupNormalization"(%X, %scale, %bias) {num_groups = 4 : si64}
        : (tensor<1x8x4x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x8x4x4xf32>
    return %Y : tensor<1x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @group_normalization_per_group
  // CHECK: onnx.GroupNormalization
  // CHECK-NOT: hip.group_norm

  func.func @group_normalization_groups_eq_channels(%X: tensor<1x8x4x4xf32>,
                                                    %scale: tensor<8xf32>,
                                                    %bias: tensor<8xf32>) -> tensor<1x8x4x4xf32> {
    %Y = "onnx.GroupNormalization"(%X, %scale, %bias) {num_groups = 8 : si64}
        : (tensor<1x8x4x4xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x8x4x4xf32>
    return %Y : tensor<1x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @group_normalization_groups_eq_channels
  // CHECK-NOT: onnx.GroupNormalization
  // CHECK: hip.group_norm
  // CHECK-SAME: groups = 8
}
