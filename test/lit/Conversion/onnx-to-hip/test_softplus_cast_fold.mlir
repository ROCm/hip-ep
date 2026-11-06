// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering fold of Cast(f16->f32) -> Softplus -> Cast(f32->f16)
// into an f16 hip.softplus (the Laguna attention-gate layout). An input Cast
// with other users stays; an f32 Softplus without the output Cast is
// unchanged.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1xf16>) -> tensor<1x1xf16> {
    return %arg0 : tensor<1x1xf16>
  }

  func.func @fold(%x: tensor<1x1x48xf16>) -> tensor<1x1x48xf16> {
    %0 = "onnx.Cast"(%x) {to = 1 : si64} : (tensor<1x1x48xf16>) -> tensor<1x1x48xf32>
    %1 = "onnx.Softplus"(%0) : (tensor<1x1x48xf32>) -> tensor<1x1x48xf32>
    %2 = "onnx.Cast"(%1) {to = 10 : si64} : (tensor<1x1x48xf32>) -> tensor<1x1x48xf16>
    return %2 : tensor<1x1x48xf16>
  }

  // CHECK-LABEL: func.func @fold
  // CHECK-NOT: hip.cast
  // CHECK: hip.softplus(%{{.*}}) ins(%{{.*}} : tensor<1x1x48xf16>) outs(%{{.*}} : tensor<1x1x48xf16>)
  // CHECK-NOT: hip.cast
  // CHECK: return

  // The f32 input is also returned: its Cast stays, the Softplus still folds.
  func.func @input_shared(%x: tensor<1x1x48xf16>)
      -> (tensor<1x1x48xf16>, tensor<1x1x48xf32>) {
    %0 = "onnx.Cast"(%x) {to = 1 : si64} : (tensor<1x1x48xf16>) -> tensor<1x1x48xf32>
    %1 = "onnx.Softplus"(%0) : (tensor<1x1x48xf32>) -> tensor<1x1x48xf32>
    %2 = "onnx.Cast"(%1) {to = 10 : si64} : (tensor<1x1x48xf32>) -> tensor<1x1x48xf16>
    return %2, %0 : tensor<1x1x48xf16>, tensor<1x1x48xf32>
  }

  // CHECK-LABEL: func.func @input_shared
  // CHECK: hip.cast
  // CHECK-NOT: hip.cast
  // CHECK: hip.softplus(%{{.*}}) ins(%{{.*}} : tensor<1x1x48xf16>) outs(%{{.*}} : tensor<1x1x48xf16>)
  // CHECK-NOT: hip.cast
  // CHECK: return

  // No output Cast: the f32 Softplus is kept.
  func.func @no_output_cast(%x: tensor<1x1x48xf16>) -> tensor<1x1x48xf32> {
    %0 = "onnx.Cast"(%x) {to = 1 : si64} : (tensor<1x1x48xf16>) -> tensor<1x1x48xf32>
    %1 = "onnx.Softplus"(%0) : (tensor<1x1x48xf32>) -> tensor<1x1x48xf32>
    return %1 : tensor<1x1x48xf32>
  }

  // CHECK-LABEL: func.func @no_output_cast
  // CHECK: hip.cast
  // CHECK: hip.softplus(%{{.*}}) ins(%{{.*}} : tensor<1x1x48xf32>) outs(%{{.*}} : tensor<1x1x48xf32>)
  // CHECK: return
}
