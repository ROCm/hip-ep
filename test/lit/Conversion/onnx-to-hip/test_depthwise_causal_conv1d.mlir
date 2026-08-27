// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A rank-3 depthwise onnx.Conv with left-only padding of exactly k-1 IS a
// causal convolution with a zero carry state, so it routes to
// hip.causal_conv_with_state (one fused kernel, bias included) instead of the
// generic hip.conv -> MIOpen path. past_state is absent, which the runtime
// kernel reads as a zero carry -- the definition of pads=[k-1, 0].
//
// The gemma-4 E2B/E4B audio encoders contain 12 of these each: k=5, C=1024,
// dynamic batch and length.
//
// Also asserts the guards, since DepthwiseCausalConvToHip runs at benefit 2 and
// would otherwise silently swallow convolutions it computes incorrectly:
// non-causal pads, stride != 1, dilation != 1, group != C, and k > 8 all have
// to fall through to hip.conv.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x1024x750xf16>) -> tensor<1x1024x750xf16> {
    return %arg0 : tensor<1x1024x750xf16>
  }

  // --------------------------------------------------------------------------
  // The audio-encoder shape, static.
  // --------------------------------------------------------------------------
  func.func @causal_dw_static(%input: tensor<1x1024x750xf16>,
                              %weights: tensor<1024x1x5xf16>,
                              %bias: tensor<1024xf16>) -> tensor<1x1024x750xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5],
      strides = [1],
      pads = [4, 0],
      group = 1024 : i64
    } : (tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x750xf16>
    return %output : tensor<1x1024x750xf16>
  }

  // CHECK-LABEL: func.func @causal_dw_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x1024x750xf16>, %[[W:.*]]: tensor<1024x1x5xf16>, %[[B:.*]]: tensor<1024xf16>)
  // CHECK-DAG: %[[OUT:.*]] = tensor.empty() : tensor<1x1024x750xf16>
  // The carry state is [B, C, k-1] and nothing reads it; ONNX Conv is stateless.
  // CHECK-DAG: %[[ST:.*]] = tensor.empty() : tensor<1x1024x4xf16>
  // activation="none", ndim=1 and channels_last=false are the op's declared
  // defaults, so the printer elides them from the attr-dict.
  // CHECK: hip.causal_conv_with_state(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>) outs(%[[OUT]], %[[ST]] : tensor<1x1024x750xf16>, tensor<1x1024x4xf16>)
  // CHECK-NOT: hip.conv(

  // --------------------------------------------------------------------------
  // The shape the audio encoders actually export: dynamic batch and length.
  // The output aliases the input positionally, so both extents come off the
  // input rather than through the conv output formula.
  // --------------------------------------------------------------------------
  func.func @causal_dw_dynamic(%input: tensor<?x1024x?xf16>,
                               %weights: tensor<1024x1x5xf16>,
                               %bias: tensor<1024xf16>) -> tensor<?x1024x?xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5],
      strides = [1],
      pads = [4, 0],
      group = 1024 : i64
    } : (tensor<?x1024x?xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<?x1024x?xf16>
    return %output : tensor<?x1024x?xf16>
  }

  // CHECK-LABEL: func.func @causal_dw_dynamic
  // CHECK: tensor.empty({{.*}}) : tensor<?x1024x?xf16>
  // CHECK: tensor.empty({{.*}}) : tensor<?x1024x4xf16>
  // CHECK: hip.causal_conv_with_state
  // CHECK-NOT: hip.conv(

  // --------------------------------------------------------------------------
  // No bias: the operand is simply omitted from the fused op.
  // --------------------------------------------------------------------------
  func.func @causal_dw_no_bias(%input: tensor<1x64x128xf16>,
                               %weights: tensor<64x1x4xf16>) -> tensor<1x64x128xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %output = "onnx.Conv"(%input, %weights, %none) {
      kernel_shape = [4],
      strides = [1],
      pads = [3, 0],
      group = 64 : i64
    } : (tensor<1x64x128xf16>, tensor<64x1x4xf16>, none)
      -> tensor<1x64x128xf16>
    return %output : tensor<1x64x128xf16>
  }

  // CHECK-LABEL: func.func @causal_dw_no_bias
  // CHECK: hip.causal_conv_with_state({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x64x128xf16>, tensor<64x1x4xf16>) outs(%{{.*}}, %{{.*}} : tensor<1x64x128xf16>, tensor<1x64x3xf16>)
  // CHECK-NOT: hip.conv(

  // --------------------------------------------------------------------------
  // fp32 is on the envelope too (element_size_bytes == 4).
  // --------------------------------------------------------------------------
  func.func @causal_dw_f32(%input: tensor<2x32x64xf32>,
                           %weights: tensor<32x1x3xf32>,
                           %bias: tensor<32xf32>) -> tensor<2x32x64xf32> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3],
      strides = [1],
      pads = [2, 0],
      group = 32 : i64
    } : (tensor<2x32x64xf32>, tensor<32x1x3xf32>, tensor<32xf32>)
      -> tensor<2x32x64xf32>
    return %output : tensor<2x32x64xf32>
  }

  // CHECK-LABEL: func.func @causal_dw_f32
  // CHECK: hip.causal_conv_with_state
  // CHECK-NOT: hip.conv(

  // ==========================================================================
  // Guards: everything below must stay on hip.conv.
  // ==========================================================================

  // Right-padded, so output[t] reads future taps -- not causal.
  func.func @guard_pads_symmetric(%input: tensor<1x1024x750xf16>,
                                  %weights: tensor<1024x1x5xf16>,
                                  %bias: tensor<1024xf16>) -> tensor<1x1024x750xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5], strides = [1], pads = [2, 2], group = 1024 : i64
    } : (tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x750xf16>
    return %output : tensor<1x1024x750xf16>
  }

  // CHECK-LABEL: func.func @guard_pads_symmetric
  // CHECK: hip.conv(
  // CHECK-NOT: hip.causal_conv_with_state

  // Stride 2: the fused kernel emits one output per input position.
  func.func @guard_stride2(%input: tensor<1x1024x750xf16>,
                           %weights: tensor<1024x1x5xf16>,
                           %bias: tensor<1024xf16>) -> tensor<1x1024x375xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5], strides = [2], pads = [4, 0], group = 1024 : i64
    } : (tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x375xf16>
    return %output : tensor<1x1024x375xf16>
  }

  // CHECK-LABEL: func.func @guard_stride2
  // CHECK: hip.conv(
  // CHECK-NOT: hip.causal_conv_with_state

  // Dilation 2: the taps are not contiguous. ConvToHip declines a dilated 1D
  // conv too, so this one stays an unconverted onnx.Conv -- the point here is
  // only that the causal route does not claim it.
  func.func @guard_dilation2(%input: tensor<1x1024x750xf16>,
                             %weights: tensor<1024x1x5xf16>,
                             %bias: tensor<1024xf16>) -> tensor<1x1024x742xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5], strides = [1], pads = [4, 0], dilations = [2],
      group = 1024 : i64
    } : (tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x742xf16>
    return %output : tensor<1x1024x742xf16>
  }

  // CHECK-LABEL: func.func @guard_dilation2
  // CHECK-NOT: hip.causal_conv_with_state

  // group != C: a grouped, not depthwise, convolution.
  func.func @guard_grouped(%input: tensor<1x64x128xf16>,
                           %weights: tensor<64x2x4xf16>,
                           %bias: tensor<64xf16>) -> tensor<1x64x128xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [4], strides = [1], pads = [3, 0], group = 32 : i64
    } : (tensor<1x64x128xf16>, tensor<64x2x4xf16>, tensor<64xf16>)
      -> tensor<1x64x128xf16>
    return %output : tensor<1x64x128xf16>
  }

  // CHECK-LABEL: func.func @guard_grouped
  // CHECK: hip.conv(
  // CHECK-NOT: hip.causal_conv_with_state

  // k = 9 is past the fused kernel's [2,8] template instantiations.
  func.func @guard_k9(%input: tensor<1x64x128xf16>,
                      %weights: tensor<64x1x9xf16>,
                      %bias: tensor<64xf16>) -> tensor<1x64x128xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [9], strides = [1], pads = [8, 0], group = 64 : i64
    } : (tensor<1x64x128xf16>, tensor<64x1x9xf16>, tensor<64xf16>)
      -> tensor<1x64x128xf16>
    return %output : tensor<1x64x128xf16>
  }

  // CHECK-LABEL: func.func @guard_k9
  // CHECK: hip.conv(
  // CHECK-NOT: hip.causal_conv_with_state

  // k = 1 would need a zero-extent [B,C,0] carry state for what is only a
  // per-channel scale.
  func.func @guard_k1(%input: tensor<1x64x128xf16>,
                      %weights: tensor<64x1x1xf16>,
                      %bias: tensor<64xf16>) -> tensor<1x64x128xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [1], strides = [1], pads = [0, 0], group = 64 : i64
    } : (tensor<1x64x128xf16>, tensor<64x1x1xf16>, tensor<64xf16>)
      -> tensor<1x64x128xf16>
    return %output : tensor<1x64x128xf16>
  }

  // CHECK-LABEL: func.func @guard_k1
  // CHECK: hip.conv(
  // CHECK-NOT: hip.causal_conv_with_state
}
