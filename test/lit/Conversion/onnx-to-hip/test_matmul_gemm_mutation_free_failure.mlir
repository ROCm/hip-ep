// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: tensor.empty
  // CHECK: "onnx.MatMul"
  func.func @main_graph(
      %a: tensor<?x4x8xf16>, %b: tensor<8x16xf16>)
      -> tensor<?x4x32xf16> {
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<?x4x8xf16>, tensor<8x16xf16>) -> tensor<?x4x32xf16>
    return %result : tensor<?x4x32xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: tensor.empty
  // CHECK: "onnx.Gemm"
  func.func @main_graph(
      %a: tensor<?x8xf16>, %b: tensor<8x16xf16>)
      -> tensor<?x32xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Gemm"(%a, %b, %none)
      : (tensor<?x8xf16>, tensor<8x16xf16>, none) -> tensor<?x32xf16>
    return %result : tensor<?x32xf16>
  }
}
