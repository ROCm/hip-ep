// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify rank-3 onnx.Conv (1D conv) is lowered to hip.conv1d. Whisper-large-v3
// encoder uses two such convs in its mel-spectrogram front-end:
//   layer 0: [1, 128, 3000] @ k=3, s=1, pad=1 -> [1, 1280, 3000]
//   layer 1: [1, 1280, 3000] @ k=3, s=2, pad=1 -> [1, 1280, 1500]
//
// Asserts:
// - hip.conv1d (NOT hip.conv) is emitted for rank-3 input
// - kernel_shape / strides / pads attributes are forwarded (single-element
//   for kernel_shape + strides; two-element [begin, end] for pads)
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
  // CHECK: tensor.empty() : tensor<1x1280x3000xf16>
  // CHECK: hip.conv1d(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x128x3000xf16>, tensor<1280x128x3xf16>, tensor<1280xf16>) outs({{.*}} : tensor<1x1280x3000xf16>) {kernel_shape = [3], pads = [1, 1], strides = [1]}
  // CHECK-NOT: hip.conv(
  // CHECK-NOT: hip.alloc

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
  // CHECK: tensor.empty() : tensor<1x1280x1500xf16>
  // CHECK: hip.conv1d(%[[CTX]]) ins(%[[IN]], %[[W]], %[[B]] : tensor<1x1280x3000xf16>, tensor<1280x1280x3xf16>, tensor<1280xf16>) outs({{.*}} : tensor<1x1280x1500xf16>) {kernel_shape = [3], pads = [1, 1], strides = [2]}
  // CHECK-NOT: hip.conv(
  // CHECK-NOT: hip.alloc
}
