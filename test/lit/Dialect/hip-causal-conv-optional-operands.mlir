// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | hip-mlir-opt | FileCheck %s

// What this file tests
// --------------------
// hip.causal_conv_with_state's two optional operands, bias and past_state,
// print and parse back as themselves whichever of them is present. Written as
// a bare run of `,` operands, an op holding only past_state printed as a third
// input and parsed back as bias.

// CHECK-LABEL: func.func @past_state_only
// CHECK-SAME: %[[PAST:arg[0-9]+]]: tensor<1x4x3xf32>
// CHECK: hip.causal_conv_with_state
// CHECK-SAME: tensor<4x1x4xf32>) past_state(%[[PAST]] : tensor<1x4x3xf32>) outs(
func.func @past_state_only(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %past: tensor<1x4x3xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x8xf32>, tensor<4x1x4xf32>)
      past_state(%past : tensor<1x4x3xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}

// CHECK-LABEL: func.func @bias_only
// CHECK-SAME: %[[BIAS:arg[0-9]+]]: tensor<4xf32>
// CHECK: hip.causal_conv_with_state
// CHECK-SAME: tensor<4x1x4xf32>) bias(%[[BIAS]] : tensor<4xf32>) outs(
func.func @bias_only(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %bias: tensor<4xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x8xf32>, tensor<4x1x4xf32>)
      bias(%bias : tensor<4xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}

// CHECK-LABEL: func.func @both
// CHECK-SAME: %[[BIAS:arg[0-9]+]]: tensor<4xf32>, %[[PAST:arg[0-9]+]]: tensor<1x4x3xf32>
// CHECK: hip.causal_conv_with_state
// CHECK-SAME: bias(%[[BIAS]] : tensor<4xf32>) past_state(%[[PAST]] : tensor<1x4x3xf32>)
func.func @both(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %bias: tensor<4xf32>, %past: tensor<1x4x3xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x8xf32>, tensor<4x1x4xf32>)
      bias(%bias : tensor<4xf32>)
      past_state(%past : tensor<1x4x3xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}
