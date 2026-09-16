// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// ============================================================================
// TEST PURPOSE:
// The gemma-4 E2B/E4B audio encoders each contain 12 copies of this subgraph:
// a channels-last activation is transposed to channels-first, run through a
// depthwise causal onnx.Conv (k=5, C=1024, pads=[4,0], group=C), then
// transposed back.
//
// Through the full pipeline that must become ONE runtime call to
// wrap_causal_conv_with_state -- not a conv plus a bias pass plus two
// transposes:
//   1. convert-onnx-to-hip     onnx.Conv -> hip.causal_conv_with_state
//                              (DepthwiseCausalConvToHip, benefit 2, beats the
//                              generic ConvToHip)
//   2. canonicalize            FoldTransposePairIntoChannelsLast absorbs both
//                              hip.transpose ops into channels_last
//   3. convert-hip-to-llvm     -> llvm.call @wrap_causal_conv_with_state
//
// The negative assertions are the point of the test: forward Conv must not
// reach MIOpen, and neither transpose may survive.
// ============================================================================

// CHECK: llvm.func @wrap_causal_conv_with_state
// CHECK-NOT: llvm.func @wrap_miopenConvolutionForward
// CHECK-NOT: llvm.func @wrap_transpose
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK-NOT: onnx.Conv
module {
  func.func @main_graph(
      %arg0: tensor<1x750x1024xf16> {onnx.name = "hidden"},
      %arg1: tensor<1024x1x5xf16> {onnx.name = "weight"},
      %arg2: tensor<1024xf16> {onnx.name = "bias"})
      -> (tensor<1x750x1024xf16> {onnx.name = "output"}) {
    %nlc_to_ncl = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]}
      : (tensor<1x750x1024xf16>) -> tensor<1x1024x750xf16>
    %conv = "onnx.Conv"(%nlc_to_ncl, %arg1, %arg2) {
      kernel_shape = [5],
      strides = [1],
      pads = [4, 0],
      group = 1024 : i64
    } : (tensor<1x1024x750xf16>, tensor<1024x1x5xf16>, tensor<1024xf16>)
      -> tensor<1x1024x750xf16>
    %ncl_to_nlc = "onnx.Transpose"(%conv) {perm = [0, 2, 1]}
      : (tensor<1x1024x750xf16>) -> tensor<1x750x1024xf16>
    "onnx.Return"(%ncl_to_nlc) : (tensor<1x750x1024xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
