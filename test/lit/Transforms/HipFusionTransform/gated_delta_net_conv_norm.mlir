// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The linear-attention core (conv of the leading projection columns, f32
// casts of the decay and beta columns, gated delta rule, head collapse and
// sigmoid-gated norm with the trailing projection columns) becomes one
// hip.gated_delta_net_conv_norm whose first three results replace the norm
// output and both carried states.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_channels_last
// CHECK-SAME: %[[PROJ:[a-z0-9_]+]]: tensor<1x1x768xf16>, %[[GATES:[a-z0-9_]+]]: tensor<1x1x4xf16>
// CHECK-NOT: hip.causal_conv_with_state
// CHECK-NOT: hip.cast
// CHECK: %[[R:.*]]:9 = hip.gated_delta_net_conv_norm(%{{[a-z0-9_]+}}) ins(%[[PROJ]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[GATES]],
// CHECK-SAME: beta_column = 2 : i64, conv_channels = 512 : i64, decay_column = 0 : i64, epsilon = 9.99999997E-7 : f32
// CHECK-SAME: z_column = 512 : i64
// CHECK-NOT: hip.gated_delta_net(
// CHECK-NOT: hip.gated_rms_norm
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2
func.func @fuse_channels_last(%ctx: !hip.context, %proj: tensor<1x1x768xf16>,
    %gates: tensor<1x1x4xf16>, %w: tensor<512x1x4xf16>,
    %bias: tensor<512xf16>, %past: tensor<1x512x3xf16>,
    %S: tensor<1x2x128x128xf32>, %A: tensor<2xf32>, %dt: tensor<2xf32>,
    %nw: tensor<128xf16>)
    -> (tensor<1x1x256xf16>, tensor<1x512x3xf16>, tensor<1x2x128x128xf32>) {
  %x = tensor.extract_slice %proj[0, 0, 0] [1, 1, 512] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x512xf16>
  %e0 = tensor.empty() : tensor<1x1x512xf16>
  %e1 = tensor.empty() : tensor<1x512x3xf16>
  %c:2 = hip.causal_conv_with_state(%ctx) ins(%x, %w, %bias, %past
      : tensor<1x1x512xf16>, tensor<512x1x4xf16>, tensor<512xf16>,
        tensor<1x512x3xf16>)
      outs(%e0, %e1 : tensor<1x1x512xf16>, tensor<1x512x3xf16>)
      {activation = "silu", channels_last = true}
      : tensor<1x1x512xf16>, tensor<1x512x3xf16>
  %ds = tensor.extract_slice %gates[0, 0, 0] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e2 = tensor.empty() : tensor<1x1x2xf32>
  %d = hip.cast(%ctx) ins(%ds : tensor<1x1x2xf16>)
      outs(%e2 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %bs = tensor.extract_slice %gates[0, 0, 2] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e3 = tensor.empty() : tensor<1x1x2xf32>
  %b = hip.cast(%ctx) ins(%bs : tensor<1x1x2xf16>)
      outs(%e3 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %e4 = tensor.empty() : tensor<1x1x2x128xf16>
  %e5 = tensor.empty() : tensor<1x2x128x128xf32>
  %o:2 = hip.gated_delta_net(%ctx) ins(%c#0, %d, %b, %S, %A, %dt
      : tensor<1x1x512xf16>, tensor<1x1x2xf32>, tensor<1x1x2xf32>,
        tensor<1x2x128x128xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%e4, %e5 : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>)
      : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>
  %f = tensor.collapse_shape %o#0 [[0], [1], [2, 3]]
      : tensor<1x1x2x128xf16> into tensor<1x1x256xf16>
  %z = tensor.extract_slice %proj[0, 0, 512] [1, 1, 256] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x256xf16>
  %e6 = tensor.empty() : tensor<1x1x256xf16>
  %y = hip.gated_rms_norm(%ctx) ins(%f, %nw, %z
      : tensor<1x1x256xf16>, tensor<128xf16>, tensor<1x1x256xf16>)
      outs(%e6 : tensor<1x1x256xf16>)
      {activation = "sigmoid", epsilon = 9.99999997E-7 : f32}
      : tensor<1x1x256xf16>
  return %y, %c#1, %o#1
      : tensor<1x1x256xf16>, tensor<1x512x3xf16>, tensor<1x2x128x128xf32>
}

// -----

// The channels-first convolution bracketed by [0, 2, 1] transposes, with
// dynamic batch and sequence axes sized through tensor.dim of every
// intermediate, as the frontend emits it.
// CHECK-LABEL: func.func @fuse_transposed
// CHECK-SAME: %[[PROJ:[a-z0-9_]+]]: tensor<?x?x768xf16>
// CHECK-NOT: hip.transpose
// CHECK-NOT: hip.causal_conv_with_state
// CHECK: %[[CIN:.*]] = tensor.empty(%[[D0:[a-z0-9_]+]], %[[D1:[a-z0-9_]+]]) : tensor<?x?x512xf16>
// CHECK-NEXT: %[[COUT:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x512xf16>
// CHECK-NEXT: %[[Z:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x256xf16>
// CHECK: %[[R:.*]]:9 = hip.gated_delta_net_conv_norm(%{{[a-z0-9_]+}}) ins(%[[PROJ]],
// CHECK-SAME: %[[CIN]], %[[COUT]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}}, %[[Z]] :
// CHECK-NOT: hip.transpose
// CHECK-NOT: hip.gated_rms_norm
// CHECK: return %[[R]]#0, %[[R]]#1, %[[R]]#2
func.func @fuse_transposed(%ctx: !hip.context, %proj: tensor<?x?x768xf16>,
    %gates: tensor<?x?x4xf16>, %w: tensor<512x1x4xf16>,
    %bias: tensor<512xf16>, %past: tensor<?x512x3xf16>,
    %S: tensor<?x2x128x128xf32>, %A: tensor<2xf32>, %dt: tensor<2xf32>,
    %nw: tensor<128xf16>)
    -> (tensor<?x?x256xf16>, tensor<?x512x3xf16>, tensor<?x2x128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %p0 = tensor.dim %proj, %c0 : tensor<?x?x768xf16>
  %p1 = tensor.dim %proj, %c1 : tensor<?x?x768xf16>
  %x = tensor.extract_slice %proj[0, 0, 0] [%p0, %p1, 512] [1, 1, 1]
      : tensor<?x?x768xf16> to tensor<?x?x512xf16>
  %t0e = tensor.empty(%p0, %p1) : tensor<?x512x?xf16>
  %t0 = hip.transpose(%ctx) ins(%x : tensor<?x?x512xf16>)
      outs(%t0e : tensor<?x512x?xf16>) {perm = [0, 2, 1]} : tensor<?x512x?xf16>
  %t00 = tensor.dim %t0, %c0 : tensor<?x512x?xf16>
  %t02 = tensor.dim %t0, %c2 : tensor<?x512x?xf16>
  %e0 = tensor.empty(%t00, %t02) : tensor<?x512x?xf16>
  %pb = tensor.dim %past, %c0 : tensor<?x512x3xf16>
  %e1 = tensor.empty(%pb) : tensor<?x512x3xf16>
  %c:2 = hip.causal_conv_with_state(%ctx) ins(%t0, %w, %bias, %past
      : tensor<?x512x?xf16>, tensor<512x1x4xf16>, tensor<512xf16>,
        tensor<?x512x3xf16>)
      outs(%e0, %e1 : tensor<?x512x?xf16>, tensor<?x512x3xf16>)
      {activation = "silu"} : tensor<?x512x?xf16>, tensor<?x512x3xf16>
  %c00 = tensor.dim %c#0, %c0 : tensor<?x512x?xf16>
  %c02 = tensor.dim %c#0, %c2 : tensor<?x512x?xf16>
  %t1e = tensor.empty(%c00, %c02) : tensor<?x?x512xf16>
  %t1 = hip.transpose(%ctx) ins(%c#0 : tensor<?x512x?xf16>)
      outs(%t1e : tensor<?x?x512xf16>) {perm = [0, 2, 1]} : tensor<?x?x512xf16>
  %g0 = tensor.dim %gates, %c0 : tensor<?x?x4xf16>
  %g1 = tensor.dim %gates, %c1 : tensor<?x?x4xf16>
  %ds = tensor.extract_slice %gates[0, 0, 0] [%g0, %g1, 2] [1, 1, 1]
      : tensor<?x?x4xf16> to tensor<?x?x2xf16>
  %bs = tensor.extract_slice %gates[0, 0, 2] [%g0, %g1, 2] [1, 1, 1]
      : tensor<?x?x4xf16> to tensor<?x?x2xf16>
  %e2 = tensor.empty(%g0, %g1) : tensor<?x?x2xf32>
  %d = hip.cast(%ctx) ins(%ds : tensor<?x?x2xf16>)
      outs(%e2 : tensor<?x?x2xf32>) {to = 1 : i64} : tensor<?x?x2xf32>
  %e3 = tensor.empty(%g0, %g1) : tensor<?x?x2xf32>
  %b = hip.cast(%ctx) ins(%bs : tensor<?x?x2xf16>)
      outs(%e3 : tensor<?x?x2xf32>) {to = 1 : i64} : tensor<?x?x2xf32>
  %t10 = tensor.dim %t1, %c0 : tensor<?x?x512xf16>
  %t11 = tensor.dim %t1, %c1 : tensor<?x?x512xf16>
  %e4 = tensor.empty(%t10, %t11) : tensor<?x?x2x128xf16>
  %sb = tensor.dim %S, %c0 : tensor<?x2x128x128xf32>
  %e5 = tensor.empty(%sb) : tensor<?x2x128x128xf32>
  %o:2 = hip.gated_delta_net(%ctx) ins(%t1, %d, %b, %S, %A, %dt
      : tensor<?x?x512xf16>, tensor<?x?x2xf32>, tensor<?x?x2xf32>,
        tensor<?x2x128x128xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%e4, %e5 : tensor<?x?x2x128xf16>, tensor<?x2x128x128xf32>)
      : tensor<?x?x2x128xf16>, tensor<?x2x128x128xf32>
  %f = tensor.collapse_shape %o#0 [[0], [1], [2, 3]]
      : tensor<?x?x2x128xf16> into tensor<?x?x256xf16>
  %f0 = tensor.dim %f, %c0 : tensor<?x?x256xf16>
  %f1 = tensor.dim %f, %c1 : tensor<?x?x256xf16>
  %z = tensor.extract_slice %proj[0, 0, 512] [%p0, %p1, 256] [1, 1, 1]
      : tensor<?x?x768xf16> to tensor<?x?x256xf16>
  %e6 = tensor.empty(%f0, %f1) : tensor<?x?x256xf16>
  %y = hip.gated_rms_norm(%ctx) ins(%f, %nw, %z
      : tensor<?x?x256xf16>, tensor<128xf16>, tensor<?x?x256xf16>)
      outs(%e6 : tensor<?x?x256xf16>)
      {activation = "sigmoid", epsilon = 9.99999997E-7 : f32}
      : tensor<?x?x256xf16>
  return %y, %c#1, %o#1
      : tensor<?x?x256xf16>, tensor<?x512x3xf16>, tensor<?x2x128x128xf32>
}

// -----

// The recurrence output is also returned, so it must still be computed: no
// fusion.
// CHECK-LABEL: func.func @reject_extra_use
// CHECK: hip.causal_conv_with_state
// CHECK: hip.gated_delta_net(
// CHECK: hip.gated_rms_norm
// CHECK-NOT: hip.gated_delta_net_conv_norm
func.func @reject_extra_use(%ctx: !hip.context, %proj: tensor<1x1x768xf16>,
    %gates: tensor<1x1x4xf16>, %w: tensor<512x1x4xf16>,
    %bias: tensor<512xf16>, %past: tensor<1x512x3xf16>,
    %S: tensor<1x2x128x128xf32>, %A: tensor<2xf32>, %dt: tensor<2xf32>,
    %nw: tensor<128xf16>)
    -> (tensor<1x1x256xf16>, tensor<1x1x2x128xf16>) {
  %x = tensor.extract_slice %proj[0, 0, 0] [1, 1, 512] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x512xf16>
  %e0 = tensor.empty() : tensor<1x1x512xf16>
  %e1 = tensor.empty() : tensor<1x512x3xf16>
  %c:2 = hip.causal_conv_with_state(%ctx) ins(%x, %w, %bias, %past
      : tensor<1x1x512xf16>, tensor<512x1x4xf16>, tensor<512xf16>,
        tensor<1x512x3xf16>)
      outs(%e0, %e1 : tensor<1x1x512xf16>, tensor<1x512x3xf16>)
      {activation = "silu", channels_last = true}
      : tensor<1x1x512xf16>, tensor<1x512x3xf16>
  %ds = tensor.extract_slice %gates[0, 0, 0] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e2 = tensor.empty() : tensor<1x1x2xf32>
  %d = hip.cast(%ctx) ins(%ds : tensor<1x1x2xf16>)
      outs(%e2 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %bs = tensor.extract_slice %gates[0, 0, 2] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e3 = tensor.empty() : tensor<1x1x2xf32>
  %b = hip.cast(%ctx) ins(%bs : tensor<1x1x2xf16>)
      outs(%e3 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %e4 = tensor.empty() : tensor<1x1x2x128xf16>
  %e5 = tensor.empty() : tensor<1x2x128x128xf32>
  %o:2 = hip.gated_delta_net(%ctx) ins(%c#0, %d, %b, %S, %A, %dt
      : tensor<1x1x512xf16>, tensor<1x1x2xf32>, tensor<1x1x2xf32>,
        tensor<1x2x128x128xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%e4, %e5 : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>)
      : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>
  %f = tensor.collapse_shape %o#0 [[0], [1], [2, 3]]
      : tensor<1x1x2x128xf16> into tensor<1x1x256xf16>
  %z = tensor.extract_slice %proj[0, 0, 512] [1, 1, 256] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x256xf16>
  %e6 = tensor.empty() : tensor<1x1x256xf16>
  %y = hip.gated_rms_norm(%ctx) ins(%f, %nw, %z
      : tensor<1x1x256xf16>, tensor<128xf16>, tensor<1x1x256xf16>)
      outs(%e6 : tensor<1x1x256xf16>)
      {activation = "sigmoid", epsilon = 9.99999997E-7 : f32}
      : tensor<1x1x256xf16>
  return %y, %o#0 : tensor<1x1x256xf16>, tensor<1x1x2x128xf16>
}

// -----

// A SiLU-gated norm is a different operator: no fusion.
// CHECK-LABEL: func.func @reject_silu_gate
// CHECK-NOT: hip.gated_delta_net_conv_norm
// CHECK: hip.gated_rms_norm
func.func @reject_silu_gate(%ctx: !hip.context, %proj: tensor<1x1x768xf16>,
    %gates: tensor<1x1x4xf16>, %w: tensor<512x1x4xf16>,
    %bias: tensor<512xf16>, %past: tensor<1x512x3xf16>,
    %S: tensor<1x2x128x128xf32>, %A: tensor<2xf32>, %dt: tensor<2xf32>,
    %nw: tensor<128xf16>)
    -> (tensor<1x1x256xf16>, tensor<1x512x3xf16>, tensor<1x2x128x128xf32>) {
  %x = tensor.extract_slice %proj[0, 0, 0] [1, 1, 512] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x512xf16>
  %e0 = tensor.empty() : tensor<1x1x512xf16>
  %e1 = tensor.empty() : tensor<1x512x3xf16>
  %c:2 = hip.causal_conv_with_state(%ctx) ins(%x, %w, %bias, %past
      : tensor<1x1x512xf16>, tensor<512x1x4xf16>, tensor<512xf16>,
        tensor<1x512x3xf16>)
      outs(%e0, %e1 : tensor<1x1x512xf16>, tensor<1x512x3xf16>)
      {activation = "silu", channels_last = true}
      : tensor<1x1x512xf16>, tensor<1x512x3xf16>
  %ds = tensor.extract_slice %gates[0, 0, 0] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e2 = tensor.empty() : tensor<1x1x2xf32>
  %d = hip.cast(%ctx) ins(%ds : tensor<1x1x2xf16>)
      outs(%e2 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %bs = tensor.extract_slice %gates[0, 0, 2] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x4xf16> to tensor<1x1x2xf16>
  %e3 = tensor.empty() : tensor<1x1x2xf32>
  %b = hip.cast(%ctx) ins(%bs : tensor<1x1x2xf16>)
      outs(%e3 : tensor<1x1x2xf32>) {to = 1 : i64} : tensor<1x1x2xf32>
  %e4 = tensor.empty() : tensor<1x1x2x128xf16>
  %e5 = tensor.empty() : tensor<1x2x128x128xf32>
  %o:2 = hip.gated_delta_net(%ctx) ins(%c#0, %d, %b, %S, %A, %dt
      : tensor<1x1x512xf16>, tensor<1x1x2xf32>, tensor<1x1x2xf32>,
        tensor<1x2x128x128xf32>, tensor<2xf32>, tensor<2xf32>)
      outs(%e4, %e5 : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>)
      : tensor<1x1x2x128xf16>, tensor<1x2x128x128xf32>
  %f = tensor.collapse_shape %o#0 [[0], [1], [2, 3]]
      : tensor<1x1x2x128xf16> into tensor<1x1x256xf16>
  %z = tensor.extract_slice %proj[0, 0, 512] [1, 1, 256] [1, 1, 1]
      : tensor<1x1x768xf16> to tensor<1x1x256xf16>
  %e6 = tensor.empty() : tensor<1x1x256xf16>
  %y = hip.gated_rms_norm(%ctx) ins(%f, %nw, %z
      : tensor<1x1x256xf16>, tensor<128xf16>, tensor<1x1x256xf16>)
      outs(%e6 : tensor<1x1x256xf16>)
      {activation = "silu", epsilon = 9.99999997E-7 : f32}
      : tensor<1x1x256xf16>
  return %y, %c#1, %o#1
      : tensor<1x1x256xf16>, tensor<1x512x3xf16>, tensor<1x2x128x128xf32>
}
