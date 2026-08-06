// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ReduceL2 E2E full pipeline: onnx.ReduceL2 -> hip.reduce_l2 -> wrap_reduce_l2
// Pattern matches SwinV2 attention Q/K L2 normalization (axis=-1, keepdims=1).
// ============================================================================

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<128x3x256x32xf16>, %arg1: tensor<1xi64>) -> (tensor<128x3x256x1xf16>) {
    // CHECK: llvm.func @wrap_reduce_l2
    // CHECK-NOT: onnx.ReduceL2

    %0 = "onnx.ReduceL2"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<128x3x256x32xf16>, tensor<1xi64>) -> tensor<128x3x256x1xf16>
    return %0 : tensor<128x3x256x1xf16>
  }
}
