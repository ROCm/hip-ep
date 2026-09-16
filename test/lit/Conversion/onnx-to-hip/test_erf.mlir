// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @erf_f32(%input: tensor<1x128x200x200xf32>) -> tensor<1x128x200x200xf32> {
    %result = "onnx.Erf"(%input) : (tensor<1x128x200x200xf32>) -> tensor<1x128x200x200xf32>
    return %result : tensor<1x128x200x200xf32>
  }

  // CHECK-LABEL: func.func @erf_f32
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x128x200x200xf32>)
  // CHECK-NOT: onnx.Erf
  // CHECK: tensor.empty() : tensor<1x128x200x200xf32>
  // CHECK: hip.erf(%[[CTX]]) ins(%[[IN]] : tensor<1x128x200x200xf32>)

  func.func @erf_dynamic(%input: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %result = "onnx.Erf"(%input) : (tensor<?x?xf16>) -> tensor<?x?xf16>
    return %result : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @erf_dynamic
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
  // CHECK: hip.erf(%[[CTX2]]) ins(%[[ARG]] : tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>)
}
