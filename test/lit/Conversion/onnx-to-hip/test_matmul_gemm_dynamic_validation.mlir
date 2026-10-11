// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.matmul
  func.func @main_graph(
      %a: tensor<2x?xf16>, %b: tensor<?x8xf16>) -> tensor<2x8xf16> {
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<2x?xf16>, tensor<?x8xf16>) -> tensor<2x8xf16>
    return %result : tensor<2x8xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.gemm
  func.func @main_graph(
      %a: tensor<2x?xf16>, %b: tensor<8x4xf16>) -> tensor<2x4xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Gemm"(%a, %b, %none)
      : (tensor<2x?xf16>, tensor<8x4xf16>, none) -> tensor<2x4xf16>
    return %result : tensor<2x4xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.matmul
  func.func @main_graph(
      %a: tensor<2x4xf16>, %b: tensor<?x8xf16>) -> tensor<2x8xf16> {
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<2x4xf16>, tensor<?x8xf16>) -> tensor<2x8xf16>
    return %result : tensor<2x8xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.gemm
  func.func @main_graph(
      %a: tensor<2x4xf16>, %b: tensor<?x8xf16>) -> tensor<2x8xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %result = "onnx.Gemm"(%a, %b, %none)
      : (tensor<2x4xf16>, tensor<?x8xf16>, none) -> tensor<2x8xf16>
    return %result : tensor<2x8xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %a: tensor<2x4xf16>, %b: tensor<8x16xf16>) -> tensor<2x16xf16> {
    // expected-error @below {{matmul contraction dim mismatch: A.shape[-1]=4 vs B.shape[-2]=8}}
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<2x4xf16>, tensor<8x16xf16>) -> tensor<2x16xf16>
    return %result : tensor<2x16xf16>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.matmul
  func.func @main_graph(
      %a: tensor<?x?x4x8xf16>, %b: tensor<?x?x8x16xf16>)
      -> tensor<?x?x4x16xf16> {
    %result = "onnx.MatMul"(%a, %b)
      : (tensor<?x?x4x8xf16>, tensor<?x?x8x16xf16>)
        -> tensor<?x?x4x16xf16>
    return %result : tensor<?x?x4x16xf16>
  }
}
