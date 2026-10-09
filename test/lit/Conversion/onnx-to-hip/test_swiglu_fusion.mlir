// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// Conversion lowers the exported Sigmoid and Mul nodes one-to-one. Collapsing
// them into hip.swiglu is hip-fusion-transform, so every frontend that emits
// the same HIP chain gets it. See Transforms/HipFusionTransform/swiglu.mlir.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @swiglu_canonical(%gate: tensor<8x16xf16>, %up: tensor<8x16xf16>)
      -> tensor<8x16xf16> {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %y : tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_canonical
  // CHECK: hip.sigmoid
  // CHECK: hip.mul
  // CHECK: hip.mul
  // CHECK-NOT: hip.swiglu
}
