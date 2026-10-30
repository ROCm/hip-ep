// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify onnx.Softmax converts to hip.miopen.softmax only when it normalizes
// over the last dimension, which is the only axis the runtime kernel handles.
//
// Assertions:
// - an explicit last axis (positive or negative) converts, for f16/bf16/f32
// - a missing axis converts on rank <= 2, where the opset < 13 default (1)
//   and the opset >= 13 default (-1) agree
// - a missing axis on rank > 2 and a non-last axis stay as onnx.Softmax
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<2x8x64xf16>)
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x8x64xf16>
  // CHECK: hip.miopen.softmax(%[[CTX]]) ins(%[[IN]] : tensor<2x8x64xf16>) outs(%[[INIT]] : tensor<2x8x64xf16>)
  // CHECK-NOT: onnx.Softmax
  func.func @main_graph(%x: tensor<2x8x64xf16>) -> tensor<2x8x64xf16> {
    %y = "onnx.Softmax"(%x) {axis = 2 : si64} : (tensor<2x8x64xf16>) -> tensor<2x8x64xf16>
    return %y : tensor<2x8x64xf16>
  }

  // CHECK-LABEL: func.func @axis_last_negative_bf16
  // CHECK: hip.miopen.softmax({{.*}}) ins({{.*}} : tensor<1x4x16x1500xbf16>)
  // CHECK-NOT: onnx.Softmax
  func.func @axis_last_negative_bf16(%x: tensor<1x4x16x1500xbf16>) -> tensor<1x4x16x1500xbf16> {
    %y = "onnx.Softmax"(%x) {axis = -1 : si64} : (tensor<1x4x16x1500xbf16>) -> tensor<1x4x16x1500xbf16>
    return %y : tensor<1x4x16x1500xbf16>
  }

  // CHECK-LABEL: func.func @axis_default_rank2
  // CHECK: hip.miopen.softmax({{.*}}) ins({{.*}} : tensor<1x1000xf32>)
  // CHECK-NOT: onnx.Softmax
  func.func @axis_default_rank2(%x: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %y = "onnx.Softmax"(%x) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %y : tensor<1x1000xf32>
  }

  // CHECK-LABEL: func.func @axis_dynamic_last
  // CHECK: hip.miopen.softmax({{.*}}) ins({{.*}} : tensor<?x?x4096xf16>)
  func.func @axis_dynamic_last(%x: tensor<?x?x4096xf16>) -> tensor<?x?x4096xf16> {
    %y = "onnx.Softmax"(%x) {axis = 2 : si64} : (tensor<?x?x4096xf16>) -> tensor<?x?x4096xf16>
    return %y : tensor<?x?x4096xf16>
  }

  // CHECK-LABEL: func.func @axis_not_last
  // CHECK: "onnx.Softmax"
  // CHECK-NOT: hip.miopen.softmax
  func.func @axis_not_last(%x: tensor<2x8x64xf16>) -> tensor<2x8x64xf16> {
    %y = "onnx.Softmax"(%x) {axis = 1 : si64} : (tensor<2x8x64xf16>) -> tensor<2x8x64xf16>
    return %y : tensor<2x8x64xf16>
  }

  // CHECK-LABEL: func.func @axis_default_rank3
  // CHECK: "onnx.Softmax"
  // CHECK-NOT: hip.miopen.softmax
  func.func @axis_default_rank3(%x: tensor<2x8x64xf32>) -> tensor<2x8x64xf32> {
    %y = "onnx.Softmax"(%x) : (tensor<2x8x64xf32>) -> tensor<2x8x64xf32>
    return %y : tensor<2x8x64xf32>
  }
}
