// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip-winograd-conv rewrites an eligible fp16 3x3 conv, with the
// activation that alone consumes it, into hip.winograd_conv, and leaves every
// conv that the kernel cannot run, or that the gate rejects, alone.
//
// Test cases:
// 1. leaky_relu     - conv + LeakyRelu become one op with the activation
// 2. relu           - conv + max(x, 0) become one op with "relu"
// 3. bias_only      - a conv with no activation user keeps "none"
// 4. memory_filter  - a memory-address filter is a constant too
// 5. two_users      - the activation is not folded when the conv has 2 users
// 6. stride2        - stride 2 is not eligible
// 7. f32            - f32 is not eligible
// 8. big_small      - 512 -> 512 channels at 16x16 is gated out
// 9. view_input     - a conv that reads a transpose is left to rocMLIR
// 10. computed_filter - a filter that is not a constant is not eligible
// ============================================================================

// RUN: hip-mlir-opt %s --split-input-file --hip-winograd-conv | FileCheck %s

// CHECK-LABEL: func.func @leaky_relu
// CHECK-NOT: hip.conv
// CHECK-NOT: hip.leaky_relu
// CHECK: %[[Y:.*]] = hip.winograd_conv(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x64x72x84xf16>, tensor<64x64x3x3xf16>, tensor<64xf16>) outs(%{{.*}} : tensor<1x64x72x84xf16>) {activation = "leaky_relu", alpha = 1.{{0*}}e-01 : f64}
// CHECK: return %[[Y]]
func.func @leaky_relu(%ctx: !hip.context, %x: tensor<1x64x72x84xf16>)
    -> tensor<1x64x72x84xf16> {
  %w = hip.constant {value = dense<0.5> : tensor<64x64x3x3xf16>} : tensor<64x64x3x3xf16>
  %b = hip.constant {value = dense<1.0> : tensor<64xf16>} : tensor<64xf16>
  %i0 = tensor.empty() : tensor<1x64x72x84xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x64x72x84xf16>,
      tensor<64x64x3x3xf16>, tensor<64xf16>) outs(%i0 : tensor<1x64x72x84xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x64x72x84xf16>
  %i1 = tensor.empty() : tensor<1x64x72x84xf16>
  %y = hip.leaky_relu(%ctx) ins(%c : tensor<1x64x72x84xf16>)
      outs(%i1 : tensor<1x64x72x84xf16>) {alpha = 0.1 : f64} : tensor<1x64x72x84xf16>
  return %y : tensor<1x64x72x84xf16>
}

// -----

// CHECK-LABEL: func.func @relu
// CHECK-NOT: hip.max
// CHECK: hip.winograd_conv(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x32x64x64xf16>, tensor<32x32x3x3xf16>) outs(%{{.*}} : tensor<1x32x64x64xf16>) {activation = "relu"
func.func @relu(%ctx: !hip.context, %x: tensor<1x32x64x64xf16>)
    -> tensor<1x32x64x64xf16> {
  %w = hip.constant {value = dense<0.5> : tensor<32x32x3x3xf16>} : tensor<32x32x3x3xf16>
  %z = hip.constant {value = dense<0.0> : tensor<1xf16>} : tensor<1xf16>
  %i0 = tensor.empty() : tensor<1x32x64x64xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x32x64x64xf16>,
      tensor<32x32x3x3xf16>) outs(%i0 : tensor<1x32x64x64xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x32x64x64xf16>
  %i1 = tensor.empty() : tensor<1x32x64x64xf16>
  %y = hip.max(%ctx) ins(%c, %z : tensor<1x32x64x64xf16>, tensor<1xf16>)
      outs(%i1 : tensor<1x32x64x64xf16>) : tensor<1x32x64x64xf16>
  return %y : tensor<1x32x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @bias_only
// The default activation "none" is not printed.
// CHECK: hip.winograd_conv(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x48x32x32xf16>, tensor<48x48x3x3xf16>, tensor<48xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<1x48x32x32xf16>) : tensor<1x48x32x32xf16>
func.func @bias_only(%ctx: !hip.context, %x: tensor<1x48x32x32xf16>)
    -> tensor<1x48x32x32xf16> {
  %w = hip.constant {value = dense<0.5> : tensor<48x48x3x3xf16>} : tensor<48x48x3x3xf16>
  %b = hip.constant {value = dense<1.0> : tensor<48xf16>} : tensor<48xf16>
  %i0 = tensor.empty() : tensor<1x48x32x32xf16>
  %c = hip.conv(%ctx) ins(%x, %w, %b : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>, tensor<48xf16>) outs(%i0 : tensor<1x48x32x32xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf16>
  return %c : tensor<1x48x32x32xf16>
}

// -----

// CHECK-LABEL: func.func @memory_filter
// CHECK: hip.winograd_conv
func.func @memory_filter(%ctx: !hip.context, %x: tensor<1x48x32x32xf16>)
    -> tensor<1x48x32x32xf16> {
  %w = hip.constant {memory_address = 4096 : i64, size = 41472 : i64} : tensor<48x48x3x3xf16>
  %i0 = tensor.empty() : tensor<1x48x32x32xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>) outs(%i0 : tensor<1x48x32x32xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf16>
  return %c : tensor<1x48x32x32xf16>
}

// -----

// The conv result has a second user, so the LeakyRelu stays a separate op.
// CHECK-LABEL: func.func @two_users
// CHECK: %[[C:.*]] = hip.winograd_conv
// CHECK-SAME: outs(%{{.*}} : tensor<1x48x32x32xf16>) : tensor<1x48x32x32xf16>
// CHECK: hip.leaky_relu(%{{.*}}) ins(%[[C]]
func.func @two_users(%ctx: !hip.context, %x: tensor<1x48x32x32xf16>)
    -> (tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16>) {
  %w = hip.constant {value = dense<0.5> : tensor<48x48x3x3xf16>} : tensor<48x48x3x3xf16>
  %i0 = tensor.empty() : tensor<1x48x32x32xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>) outs(%i0 : tensor<1x48x32x32xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf16>
  %i1 = tensor.empty() : tensor<1x48x32x32xf16>
  %y = hip.leaky_relu(%ctx) ins(%c : tensor<1x48x32x32xf16>)
      outs(%i1 : tensor<1x48x32x32xf16>) {alpha = 0.1 : f64} : tensor<1x48x32x32xf16>
  return %c, %y : tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16>
}

// -----

// CHECK-LABEL: func.func @stride2
// CHECK-NOT: hip.winograd_conv
// CHECK: hip.conv
func.func @stride2(%ctx: !hip.context, %x: tensor<1x48x32x32xf16>)
    -> tensor<1x48x16x16xf16> {
  %w = hip.constant {value = dense<0.5> : tensor<48x48x3x3xf16>} : tensor<48x48x3x3xf16>
  %i0 = tensor.empty() : tensor<1x48x16x16xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>) outs(%i0 : tensor<1x48x16x16xf16>)
      {kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x16x16xf16>
  return %c : tensor<1x48x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @f32
// CHECK-NOT: hip.winograd_conv
// CHECK: hip.conv
func.func @f32(%ctx: !hip.context, %x: tensor<1x48x32x32xf32>)
    -> tensor<1x48x32x32xf32> {
  %w = hip.constant {value = dense<0.5> : tensor<48x48x3x3xf32>} : tensor<48x48x3x3xf32>
  %i0 = tensor.empty() : tensor<1x48x32x32xf32>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x48x32x32xf32>,
      tensor<48x48x3x3xf32>) outs(%i0 : tensor<1x48x32x32xf32>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf32>
  return %c : tensor<1x48x32x32xf32>
}

// -----

// Both channel counts >= 511 at a side of at most 32: the tuned rocMLIR conv
// is faster.
// CHECK-LABEL: func.func @big_small
// CHECK-NOT: hip.winograd_conv
// CHECK: hip.conv
func.func @big_small(%ctx: !hip.context, %x: tensor<1x512x16x16xf16>)
    -> tensor<1x512x16x16xf16> {
  %w = hip.constant {memory_address = 4096 : i64, size = 4718592 : i64} : tensor<512x512x3x3xf16>
  %i0 = tensor.empty() : tensor<1x512x16x16xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x512x16x16xf16>,
      tensor<512x512x3x3xf16>) outs(%i0 : tensor<1x512x16x16xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x512x16x16xf16>
  return %c : tensor<1x512x16x16xf16>
}

// -----

// CHECK-LABEL: func.func @view_input
// CHECK-NOT: hip.winograd_conv
// CHECK: hip.conv
func.func @view_input(%ctx: !hip.context, %x: tensor<1x32x32x48xf16>)
    -> tensor<1x48x32x32xf16> {
  %w = hip.constant {value = dense<0.5> : tensor<48x48x3x3xf16>} : tensor<48x48x3x3xf16>
  %it = tensor.empty() : tensor<1x48x32x32xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x32x32x48xf16>)
      outs(%it : tensor<1x48x32x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x48x32x32xf16>
  %i0 = tensor.empty() : tensor<1x48x32x32xf16>
  %c = hip.conv(%ctx) ins(%t, %w : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>) outs(%i0 : tensor<1x48x32x32xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf16>
  return %c : tensor<1x48x32x32xf16>
}

// -----

// CHECK-LABEL: func.func @computed_filter
// CHECK-NOT: hip.winograd_conv
// CHECK: hip.conv
func.func @computed_filter(%ctx: !hip.context, %x: tensor<1x48x32x32xf16>,
                           %w: tensor<48x48x3x3xf16>) -> tensor<1x48x32x32xf16> {
  %i0 = tensor.empty() : tensor<1x48x32x32xf16>
  %c = hip.conv(%ctx) ins(%x, %w : tensor<1x48x32x32xf16>,
      tensor<48x48x3x3xf16>) outs(%i0 : tensor<1x48x32x32xf16>)
      {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1],
       dilations = [1, 1], group = 1 : i64} : tensor<1x48x32x32xf16>
  return %c : tensor<1x48x32x32xf16>
}
