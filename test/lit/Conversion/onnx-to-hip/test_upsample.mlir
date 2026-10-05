// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Deeplab image-pooling upsample: 1x1 -> 65x65, scale 65, linear.
  func.func @upsample_linear_1x1_to_65(%arg0: tensor<1x256x1x1xf32>)
      -> tensor<1x256x65x65xf32> {
    // CHECK-LABEL: func.func @upsample_linear_1x1_to_65
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x256x1x1xf32>)
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 65.0, 65.0]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %y = "onnx.Upsample"(%arg0, %scales) {mode = "linear"}
        : (tensor<1x256x1x1xf32>, tensor<4xf32>) -> tensor<1x256x65x65xf32>
    // CHECK-NOT: onnx.Upsample
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x256x65x65xf32>
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<1x256x1x1xf32>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x256x65x65xf32>)
    // mode=1 (linear), coord_transform=1 (asymmetric), nearest_mode=2 (floor)
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 1
    // CHECK-SAME: nearest_mode = 2
    return %y : tensor<1x256x65x65xf32>
  }

  // Classifier upsample: 65 -> 512, scale 7.890625, linear.
  // floor(65 * 7.890625) = 512.
  func.func @upsample_linear_65_to_512(%arg0: tensor<1x21x65x65xf32>)
      -> tensor<1x21x512x512xf32> {
    // CHECK-LABEL: func.func @upsample_linear_65_to_512
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 7.890625, 7.890625]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %y = "onnx.Upsample"(%arg0, %scales) {mode = "linear"}
        : (tensor<1x21x65x65xf32>, tensor<4xf32>) -> tensor<1x21x512x512xf32>
    // CHECK-NOT: onnx.Upsample
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 1
    // CHECK-SAME: nearest_mode = 2
    return %y : tensor<1x21x512x512xf32>
  }

  // Default mode is nearest, which uses floor.
  func.func @upsample_nearest_default(%arg0: tensor<1x4x8x8xf16>)
      -> tensor<1x4x16x16xf16> {
    // CHECK-LABEL: func.func @upsample_nearest_default
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %y = "onnx.Upsample"(%arg0, %scales)
        : (tensor<1x4x8x8xf16>, tensor<4xf32>) -> tensor<1x4x16x16xf16>
    // CHECK-NOT: onnx.Upsample
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 0
    // CHECK-SAME: nearest_mode = 2
    return %y : tensor<1x4x16x16xf16>
  }

  // Rank 2 is not a spatial NCHW resize. Leave it unconverted.
  func.func @upsample_rank2(%arg0: tensor<8x8xf32>, %scales: tensor<2xf32>)
      -> tensor<16x16xf32> {
    // CHECK-LABEL: func.func @upsample_rank2
    %y = "onnx.Upsample"(%arg0, %scales) {mode = "nearest"}
        : (tensor<8x8xf32>, tensor<2xf32>) -> tensor<16x16xf32>
    // CHECK: onnx.Upsample
    // CHECK-NOT: hip.resize
    return %y : tensor<16x16xf32>
  }
}
