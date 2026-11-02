// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify onnx.TensorScatter -> hip.tensor_scatter: the default axis -2 is
// normalized to rank - 2, mode "circular" becomes the circular flag, an
// absent write_indices drops the operand, and dynamic output dims come from
// past_cache. The op stays onnx.TensorScatter when the axis is the batch
// axis or out of range, or the mode is unknown.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @default_axis
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[P:[^:]*]]: tensor<2x4x16x8xf16>, %[[U:[^:]*]]: tensor<2x4x3x8xf16>, %[[W:[^:]*]]: tensor<2xi64>)
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<2x4x16x8xf16>
  // CHECK: hip.tensor_scatter(%[[CTX]]) ins(%[[P]], %[[U]] : tensor<2x4x16x8xf16>, tensor<2x4x3x8xf16>) write_indices(%[[W]] : tensor<2xi64>) outs(%[[E]] : tensor<2x4x16x8xf16>) {axis = 2 : i64}
  // CHECK-NOT: onnx.TensorScatter
  func.func @default_axis(%past: tensor<2x4x16x8xf16>,
                          %update: tensor<2x4x3x8xf16>,
                          %wi: tensor<2xi64>) -> tensor<2x4x16x8xf16> {
    %r = "onnx.TensorScatter"(%past, %update, %wi)
        : (tensor<2x4x16x8xf16>, tensor<2x4x3x8xf16>, tensor<2xi64>) -> tensor<2x4x16x8xf16>
    return %r : tensor<2x4x16x8xf16>
  }

  // CHECK-LABEL: func.func @circular_no_indices
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[P:[^:]*]]: tensor<?x16x8xf32>
  // CHECK: %[[D0:.*]] = tensor.dim %[[P]], %{{.*}} : tensor<?x16x8xf32>
  // CHECK: %[[E:.*]] = tensor.empty(%[[D0]]) : tensor<?x16x8xf32>
  // CHECK: hip.tensor_scatter(%[[CTX]]) ins(%[[P]], %{{.*}} : tensor<?x16x8xf32>, tensor<?x2x8xf32>) outs(%[[E]] : tensor<?x16x8xf32>) {axis = 1 : i64, circular = true}
  func.func @circular_no_indices(%past: tensor<?x16x8xf32>,
                                 %update: tensor<?x2x8xf32>) -> tensor<?x16x8xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.TensorScatter"(%past, %update, %none) {axis = 1 : si64, mode = "circular"}
        : (tensor<?x16x8xf32>, tensor<?x2x8xf32>, none) -> tensor<?x16x8xf32>
    return %r : tensor<?x16x8xf32>
  }

  // CHECK-LABEL: func.func @batch_axis
  // CHECK-NOT: hip.tensor_scatter
  // CHECK: onnx.TensorScatter
  func.func @batch_axis(%past: tensor<4x16x8xf16>,
                        %update: tensor<2x16x8xf16>) -> tensor<4x16x8xf16> {
    %r = "onnx.TensorScatter"(%past, %update) {axis = 0 : si64}
        : (tensor<4x16x8xf16>, tensor<2x16x8xf16>) -> tensor<4x16x8xf16>
    return %r : tensor<4x16x8xf16>
  }

  // CHECK-LABEL: func.func @axis_out_of_range
  // CHECK-NOT: hip.tensor_scatter
  // CHECK: onnx.TensorScatter
  func.func @axis_out_of_range(%past: tensor<4x16x8xf16>,
                               %update: tensor<4x16x2xf16>) -> tensor<4x16x8xf16> {
    %r = "onnx.TensorScatter"(%past, %update) {axis = 3 : si64}
        : (tensor<4x16x8xf16>, tensor<4x16x2xf16>) -> tensor<4x16x8xf16>
    return %r : tensor<4x16x8xf16>
  }

  // CHECK-LABEL: func.func @unknown_mode
  // CHECK-NOT: hip.tensor_scatter
  // CHECK: onnx.TensorScatter
  func.func @unknown_mode(%past: tensor<4x16x8xf16>,
                          %update: tensor<4x2x8xf16>) -> tensor<4x16x8xf16> {
    %r = "onnx.TensorScatter"(%past, %update) {axis = 1 : si64, mode = "wrap"}
        : (tensor<4x16x8xf16>, tensor<4x2x8xf16>) -> tensor<4x16x8xf16>
    return %r : tensor<4x16x8xf16>
  }
}
