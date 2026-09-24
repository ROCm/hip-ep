// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-fuse-rocmlir | FileCheck %s

// MIGraphX represents scalar elementwise operands entering a rocMLIR kernel as
// one-element tensors. Verify hip-ep does the same at the outlining boundary:
// the source graph remains ONNX-compatible tensor<f16>, while the dispatch and
// outlined function use tensor<1xf16>.

// CHECK-LABEL: func.func @rocMlir0(
// CHECK-SAME: %arg3: tensor<1xf16>
// CHECK: hip.max
// CHECK-SAME: tensor<1x4x3x3xf16>, tensor<1xf16>

// CHECK-LABEL: func.func @main_graph(
// CHECK: %[[SHAPE:.*]] = arith.constant dense<1> : tensor<1xindex>
// CHECK: %[[RANK_ONE:.*]] = tensor.reshape %arg4(%[[SHAPE]]) : (tensor<f16>, tensor<1xindex>) -> tensor<1xf16>
// CHECK: hip.rocmlir
// CHECK-SAME: %[[RANK_ONE]]
// CHECK-SAME: tensor<1x4x5x5xf16>, tensor<4x4x3x3xf16>, tensor<4xf16>, tensor<1xf16>
func.func @main_graph(
    %ctx: !hip.context,
    %input: tensor<1x4x5x5xf16>,
    %weight: tensor<4x4x3x3xf16>,
    %bias: tensor<4xf16>,
    %clip_min: tensor<f16>) -> tensor<1x4x3x3xf16> {
  %conv_init = tensor.empty() : tensor<1x4x3x3xf16>
  %conv = hip.conv(%ctx) ins(
      %input, %weight, %bias :
      tensor<1x4x5x5xf16>, tensor<4x4x3x3xf16>, tensor<4xf16>)
      outs(%conv_init : tensor<1x4x3x3xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x4x3x3xf16>
  %max_init = tensor.empty() : tensor<1x4x3x3xf16>
  %max = hip.max(%ctx) ins(
      %conv, %clip_min : tensor<1x4x3x3xf16>, tensor<f16>)
      outs(%max_init : tensor<1x4x3x3xf16>) : tensor<1x4x3x3xf16>
  return %max : tensor<1x4x3x3xf16>
}
