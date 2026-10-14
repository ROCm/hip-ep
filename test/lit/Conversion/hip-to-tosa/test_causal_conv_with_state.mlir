// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// hip.causal_conv_with_state has no TOSA counterpart. Convert it to the
// Concat + depthwise Conv + Slice expansion ONNX documents, spelled with
// tosa.concat, tosa.depthwise_conv2d (length as W, H=1), and tosa.slice.
//
// FILE LAYOUT:
// Converting cases live in the first --split-input-file chunk. Each rejected
// form gets its own chunk so a legalization failure cannot mask later cases.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Concat past with the input, take the last k-1 as present_state, then a
// zero-pad depthwise conv in NHWC.
// CHECK-LABEL: func.func @causal_conv_with_past
// CHECK: %[[PADDED:.*]] = tosa.concat %arg4, %arg1 {axis = 2 : i32}
// CHECK-SAME: tensor<1x4x11xf32>
// CHECK: tosa.slice %[[PADDED]]
// CHECK-SAME: tensor<1x4x3xf32>
// CHECK: tosa.depthwise_conv2d
// CHECK-SAME: acc_type = f32
// CHECK-SAME: pad = array<i64: 0, 0, 0, 0>
// CHECK-NOT: hip.causal_conv_with_state
func.func @causal_conv_with_past(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %bias: tensor<4xf32>, %past: tensor<1x4x3xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) attributes {rock.kernel} {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight, %bias, %past :
          tensor<1x4x8xf32>, tensor<4x1x4xf32>, tensor<4xf32>,
          tensor<1x4x3xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      {activation = "none", ndim = 1 : i64}
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}

// Missing past_state is a zero tensor of length k-1.
// CHECK-LABEL: func.func @causal_conv_zero_past
// CHECK: "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x4x3xf32>}>
// CHECK: tosa.concat
// CHECK: tosa.depthwise_conv2d
// CHECK-NOT: hip.causal_conv_with_state
func.func @causal_conv_zero_past(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) attributes {rock.kernel} {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x8xf32>, tensor<4x1x4xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}

// SiLU is mul(x, sigmoid(x)) after the convolution. present_state is not
// activated.
// CHECK-LABEL: func.func @causal_conv_silu
// CHECK: tosa.depthwise_conv2d
// CHECK: tosa.sigmoid
// CHECK: tosa.mul
// CHECK-NOT: hip.causal_conv_with_state
func.func @causal_conv_silu(
    %ctx: !hip.context, %input: tensor<1x4x8xf16>, %weight: tensor<4x1x4xf16>,
    %bias: tensor<4xf16>, %past: tensor<1x4x3xf16>,
    %y_init: tensor<1x4x8xf16>, %s_init: tensor<1x4x3xf16>)
    -> (tensor<1x4x8xf16>, tensor<1x4x3xf16>) attributes {rock.kernel} {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight, %bias, %past :
          tensor<1x4x8xf16>, tensor<4x1x4xf16>, tensor<4xf16>,
          tensor<1x4x3xf16>)
      outs(%y_init, %s_init : tensor<1x4x8xf16>, tensor<1x4x3xf16>)
      {activation = "silu"}
      : tensor<1x4x8xf16>, tensor<1x4x3xf16>
  return %y, %s : tensor<1x4x8xf16>, tensor<1x4x3xf16>
}

// channels_last input/output are (B, L, C); past/present stay (B, C, k-1).
// CHECK-LABEL: func.func @causal_conv_channels_last
// CHECK: tosa.transpose %arg1 {perms = array<i32: 0, 2, 1>}
// CHECK: tosa.concat
// CHECK: tosa.depthwise_conv2d
// CHECK: tosa.transpose {{.*}} {perms = array<i32: 0, 2, 1>}
// CHECK-NOT: hip.causal_conv_with_state
func.func @causal_conv_channels_last(
    %ctx: !hip.context, %input: tensor<1x8x4xf32>, %weight: tensor<4x1x4xf32>,
    %bias: tensor<4xf32>, %past: tensor<1x4x3xf32>,
    %y_init: tensor<1x8x4xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x8x4xf32>, tensor<1x4x3xf32>) attributes {rock.kernel} {
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight, %bias, %past :
          tensor<1x8x4xf32>, tensor<4x1x4xf32>, tensor<4xf32>,
          tensor<1x4x3xf32>)
      outs(%y_init, %s_init : tensor<1x8x4xf32>, tensor<1x4x3xf32>)
      {channels_last = true}
      : tensor<1x8x4xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x8x4xf32>, tensor<1x4x3xf32>
}

// Outlined-kernel form: context is ub.poison.
// CHECK-LABEL: func.func @causal_conv_outlined
// CHECK: tosa.concat
// CHECK: tosa.depthwise_conv2d
// CHECK-NOT: hip.causal_conv_with_state
func.func @causal_conv_outlined(
    %input: tensor<1x4x8xf32>, %weight: tensor<4x1x4xf32>,
    %bias: tensor<4xf32>, %past: tensor<1x4x3xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x3xf32>) attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %y_init = tensor.empty() : tensor<1x4x8xf32>
  %s_init = tensor.empty() : tensor<1x4x3xf32>
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight, %bias, %past :
          tensor<1x4x8xf32>, tensor<4x1x4xf32>, tensor<4xf32>,
          tensor<1x4x3xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x3xf32>
}

// -----

func.func @causal_conv_dynamic(
    %ctx: !hip.context, %input: tensor<1x4x?xf32>, %weight: tensor<4x1x4xf32>,
    %y_init: tensor<1x4x?xf32>, %s_init: tensor<1x4x3xf32>)
    -> (tensor<1x4x?xf32>, tensor<1x4x3xf32>) attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.causal_conv_with_state'}}
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x?xf32>, tensor<4x1x4xf32>)
      outs(%y_init, %s_init : tensor<1x4x?xf32>, tensor<1x4x3xf32>)
      : tensor<1x4x?xf32>, tensor<1x4x3xf32>
  return %y, %s : tensor<1x4x?xf32>, tensor<1x4x3xf32>
}

// -----

// k=1 makes present_state tensor<BxCx0xT>, which TOSA rejects.
func.func @causal_conv_k1(
    %ctx: !hip.context, %input: tensor<1x4x8xf32>, %weight: tensor<4x1x1xf32>,
    %y_init: tensor<1x4x8xf32>, %s_init: tensor<1x4x0xf32>)
    -> (tensor<1x4x8xf32>, tensor<1x4x0xf32>) attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.causal_conv_with_state'}}
  %y, %s = hip.causal_conv_with_state(%ctx)
      ins(%input, %weight : tensor<1x4x8xf32>, tensor<4x1x1xf32>)
      outs(%y_init, %s_init : tensor<1x4x8xf32>, tensor<1x4x0xf32>)
      : tensor<1x4x8xf32>, tensor<1x4x0xf32>
  return %y, %s : tensor<1x4x8xf32>, tensor<1x4x0xf32>
}
