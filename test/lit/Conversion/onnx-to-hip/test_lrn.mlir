// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// onnx.LRN -> hip.lrn
//
// GoogLeNet opset 12 has two LRN nodes, both size 5, f32, rank 4:
// pool1/norm1_1 is 1x64x56x56 and conv2/norm2_1 is 1x192x56x56.
// Rank below 2 stays onnx.LRN.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }

  // CHECK-LABEL: func.func @lrn_googlenet_64
  // CHECK: hip.lrn
  // CHECK-SAME: tensor<1x64x56x56xf32>
  // CHECK-SAME: size = 5 : i64
  // CHECK-NOT: onnx.LRN
  func.func @lrn_googlenet_64(%x: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
    %y = "onnx.LRN"(%x) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32,
                         bias = 1.000000e+00 : f32, size = 5 : si64}
        : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %y : tensor<1x64x56x56xf32>
  }

  // CHECK-LABEL: func.func @lrn_googlenet_192
  // CHECK: hip.lrn
  // CHECK-SAME: tensor<1x192x56x56xf32>
  // CHECK-SAME: size = 5 : i64
  // CHECK-NOT: onnx.LRN
  func.func @lrn_googlenet_192(%x: tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32> {
    %y = "onnx.LRN"(%x) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32,
                         bias = 1.000000e+00 : f32, size = 5 : si64}
        : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
    return %y : tensor<1x192x56x56xf32>
  }

  // CHECK-LABEL: func.func @lrn_rank3
  // CHECK: hip.lrn
  // CHECK-NOT: onnx.LRN
  func.func @lrn_rank3(%x: tensor<2x8x4xf32>) -> tensor<2x8x4xf32> {
    %y = "onnx.LRN"(%x) {size = 3 : si64} : (tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
    return %y : tensor<2x8x4xf32>
  }

  // CHECK-LABEL: func.func @lrn_rank1_rejected
  // CHECK: onnx.LRN
  // CHECK-NOT: hip.lrn
  func.func @lrn_rank1_rejected(%x: tensor<8xf32>) -> tensor<8xf32> {
    %y = "onnx.LRN"(%x) {size = 5 : si64} : (tensor<8xf32>) -> tensor<8xf32>
    return %y : tensor<8xf32>
  }
}
