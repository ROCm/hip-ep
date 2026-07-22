// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify rank-3 onnx.Conv (1D conv) is lowered to the shared 2D hip.conv via a
// unit-H reshape: tensor.expand_shape (NCL -> NC1L) -> hip.conv ->
// tensor.collapse_shape (NC1L' -> NCL'). Whisper-large-v3 encoder uses two such
// convs in its mel-spectrogram front-end:
//   layer 0: [1, 128, 3000] @ k=3, s=1, pad=1 -> [1, 1280, 3000]
//   layer 1: [1, 1280, 3000] @ k=3, s=2, pad=1 -> [1, 1280, 1500]
//
// Asserts:
// - input + weights are expanded to rank-4 with a unit H dim
// - hip.conv (the shared 2D op, NOT a dedicated hip.conv1d) is emitted
// - 1D attrs are promoted to 2D H=1 form: kernel_shape [1,K], strides [1,s],
//   pads [0,begin,0,end], dilations [1,1], group 1
// - the rank-4 result is collapsed back to rank-3
// - The context arg is prepended
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<1x128x3000xf16>) -> tensor<1x128x3000xf16> {
    return %arg0 : tensor<1x128x3000xf16>
  }

  // --------------------------------------------------------------------------
  // Whisper encoder Conv layer 0: stride 1
  // --------------------------------------------------------------------------
  func.func @whisper_conv_layer0(%input: tensor<1x128x3000xf16>,
                                  %weights: tensor<1280x128x3xf16>,
                                  %bias: tensor<1280xf16>)
      -> tensor<1x1280x3000xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3],
      strides = [1],
      pads = [1, 1]
    } : (tensor<1x128x3000xf16>, tensor<1280x128x3xf16>, tensor<1280xf16>)
      -> tensor<1x1280x3000xf16>
    return %output : tensor<1x1280x3000xf16>
  }

  // CHECK-LABEL: func.func @whisper_conv_layer0
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x128x3000xf16>, %[[W:.*]]: tensor<1280x128x3xf16>, %[[B:.*]]: tensor<1280xf16>) -> tensor<1x1280x3000xf16>
  // CHECK: %[[INX:.*]] = tensor.expand_shape %[[IN]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 128, 1, 3000] : tensor<1x128x3000xf16> into tensor<1x128x1x3000xf16>
  // CHECK: %[[WX:.*]] = tensor.expand_shape %[[W]] {{\[\[}}0], [1], [2, 3]] output_shape [1280, 128, 1, 3] : tensor<1280x128x3xf16> into tensor<1280x128x1x3xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x1280x3000xf16>
  // CHECK: %[[INITX:.*]] = tensor.expand_shape %[[INIT]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 1280, 1, 3000] : tensor<1x1280x3000xf16> into tensor<1x1280x1x3000xf16>
  // CHECK: %[[CONV:.*]] = hip.conv(%[[CTX]]) ins(%[[INX]], %[[WX]], %[[B]] : tensor<1x128x1x3000xf16>, tensor<1280x128x1x3xf16>, tensor<1280xf16>) outs(%[[INITX]] : tensor<1x1280x1x3000xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 3], pads = [0, 1, 0, 1], strides = [1, 1]}
  // CHECK: tensor.collapse_shape %[[CONV]] {{\[\[}}0], [1], [2, 3]] : tensor<1x1280x1x3000xf16> into tensor<1x1280x3000xf16>
  // CHECK-NOT: hip.conv1d

  // --------------------------------------------------------------------------
  // Whisper encoder Conv layer 1: stride 2 (downsamples Lin from 3000 -> 1500)
  // --------------------------------------------------------------------------
  func.func @whisper_conv_layer1(%input: tensor<1x1280x3000xf16>,
                                  %weights: tensor<1280x1280x3xf16>,
                                  %bias: tensor<1280xf16>)
      -> tensor<1x1280x1500xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [3],
      strides = [2],
      pads = [1, 1]
    } : (tensor<1x1280x3000xf16>, tensor<1280x1280x3xf16>, tensor<1280xf16>)
      -> tensor<1x1280x1500xf16>
    return %output : tensor<1x1280x1500xf16>
  }

  // CHECK-LABEL: func.func @whisper_conv_layer1
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x1280x3000xf16>, %[[W:.*]]: tensor<1280x1280x3xf16>, %[[B:.*]]: tensor<1280xf16>) -> tensor<1x1280x1500xf16>
  // CHECK: %[[INX:.*]] = tensor.expand_shape %[[IN]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 1280, 1, 3000] : tensor<1x1280x3000xf16> into tensor<1x1280x1x3000xf16>
  // CHECK: %[[WX:.*]] = tensor.expand_shape %[[W]] {{\[\[}}0], [1], [2, 3]] output_shape [1280, 1280, 1, 3] : tensor<1280x1280x3xf16> into tensor<1280x1280x1x3xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x1280x1500xf16>
  // CHECK: %[[INITX:.*]] = tensor.expand_shape %[[INIT]] {{\[\[}}0], [1], [2, 3]] output_shape [1, 1280, 1, 1500] : tensor<1x1280x1500xf16> into tensor<1x1280x1x1500xf16>
  // CHECK: %[[CONV:.*]] = hip.conv(%[[CTX]]) ins(%[[INX]], %[[WX]], %[[B]] : tensor<1x1280x1x3000xf16>, tensor<1280x1280x1x3xf16>, tensor<1280xf16>) outs(%[[INITX]] : tensor<1x1280x1x1500xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 3], pads = [0, 1, 0, 1], strides = [1, 2]}
  // CHECK: tensor.collapse_shape %[[CONV]] {{\[\[}}0], [1], [2, 3]] : tensor<1x1280x1x1500xf16> into tensor<1x1280x1500xf16>
  // CHECK-NOT: hip.conv1d

  // --------------------------------------------------------------------------
  // Depthwise 1D conv (group = channels). Reshapes to a grouped 2D conv; the
  // `group` attr is preserved verbatim through the unit-H reshape and the
  // depthwise [C,1,K] filter expands to [C,1,1,K]. Causal pad [4,0] promotes to
  // 2D pads [0,4,0,0]. Audio-encoder lconv1d uses this shape.
  // --------------------------------------------------------------------------
  func.func @depthwise_conv1d(%input: tensor<1x1024x100xf16>,
                              %weights: tensor<1024x1x5xf16>,
                              %bias: tensor<1024xf16>) -> tensor<1x1024x100xf16> {
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [5],
      strides = [1],
      pads = [4, 0],
      group = 1024 : i64
    } : (tensor<1x1024x100xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x100xf16>
    return %output : tensor<1x1024x100xf16>
  }

  // CHECK-LABEL: func.func @depthwise_conv1d
  // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [1, 1024, 1, 100] : tensor<1x1024x100xf16> into tensor<1x1024x1x100xf16>
  // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0], [1], [2, 3]] output_shape [1024, 1, 1, 5] : tensor<1024x1x5xf16> into tensor<1024x1x1x5xf16>
  // CHECK: hip.conv({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 1024 : i64, kernel_shape = [1, 5], pads = [0, 4, 0, 0], strides = [1, 1]}
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: hip.conv1d

  // --------------------------------------------------------------------------
  // Depthwise 1D conv with dynamic batch + length. Causal padding keeps L'=L,
  // but the extent is still resolved at runtime via the conv formula (arith).
  // --------------------------------------------------------------------------
  func.func @depthwise_conv1d_dynamic(%input: tensor<?x1024x?xf16>,
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

  // CHECK-LABEL: func.func @depthwise_conv1d_dynamic
  // CHECK: arith.divsi
  // CHECK: tensor.empty({{.*}}) : tensor<?x1024x?xf16>
  // CHECK: tensor.expand_shape
  // CHECK: hip.conv({{.*}}) outs({{.*}}) {dilations = [1, 1], group = 1024 : i64, kernel_shape = [1, 5], pads = [0, 4, 0, 0], strides = [1, 1]}
  // CHECK: tensor.collapse_shape
  // CHECK-NOT: hip.conv1d
}
