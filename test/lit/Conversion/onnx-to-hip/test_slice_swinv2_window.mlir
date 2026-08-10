// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// SwinV2 shifted-window slice from /layers.0/blocks.1/Slice_1:
//   input  1x128x256x96
//   axis=1, start=0, end=8  ->  1x8x256x96
//
// Uses onnx.Constant operands (externalized by convert-onnx-to-hip) so the
// SliceShapeFold -> SliceDecompose path is exercised end-to-end.

// RUN: mkdir -p %t && env -u HIPDNN_EP_DISABLE_SLICE_DECOMPOSITION hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=1 externalize-output-dir=%t' %s | FileCheck %s
// RUN: mkdir -p %t && env HIPDNN_EP_DISABLE_SLICE_DECOMPOSITION=1 hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=1 externalize-output-dir=%t' %s | FileCheck %s --check-prefix=NO-DECOMPOSE

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @test_swinv2_window_slice(%input: tensor<1x128x256x96xf16>)
      -> tensor<1x8x256x96xf16> {
    // CHECK-LABEL: func.func @test_swinv2_window_slice
    // NO-DECOMPOSE-LABEL: func.func @test_swinv2_window_slice
    %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<8> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes)
        : (tensor<1x128x256x96xf16>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<1x8x256x96xf16>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice {{.*}}[0, 0, 0, 0] [1, 8, 256, 96] [1, 1, 1, 1]
    // NO-DECOMPOSE-NOT: tensor.extract_slice
    // NO-DECOMPOSE: hip.slice(

    return %r : tensor<1x8x256x96xf16>
  }

  // SwinV2 downsample strided slice from /layers.0/downsample/Slice_2:
  //   axis=1, start=1, end=INT64_MAX, step=2  on  1x128x256x96  ->  1x64x256x96
  func.func @test_swinv2_downsample_stride_slice(%input: tensor<1x128x256x96xf16>)
      -> tensor<1x64x256x96xf16> {
    // CHECK-LABEL: func.func @test_swinv2_downsample_stride_slice
    // NO-DECOMPOSE-LABEL: func.func @test_swinv2_downsample_stride_slice
    %starts = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<9223372036854775807> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<2> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<1x128x256x96xf16>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<1x64x256x96xf16>

    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice {{.*}}[0, 1, 0, 0] [1, 64, 256, 96] [1, 2, 1, 1]
    // NO-DECOMPOSE-NOT: tensor.extract_slice
    // NO-DECOMPOSE: hip.slice(

    return %r : tensor<1x64x256x96xf16>
  }
}
