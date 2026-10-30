// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip %s | FileCheck %s --implicit-check-not=tensor.dim --implicit-check-not=tensor.empty --implicit-check-not=arith.select --implicit-check-not=arith.cmpi

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

// A folded N conflicts with the result type after the batch merge is built.
module {
  func.func @main_graph(%a: tensor<f32>) -> tensor<f32> {
    return %a : tensor<f32>
  }
  // CHECK-LABEL: func.func @folded_matmul
  // CHECK: "onnx.MatMul"
  // CHECK: return
  func.func @folded_matmul(%n: index, %lhs: tensor<?x4x8xf32>,
                        %rhs: tensor<?x8x?xf32>) -> tensor<?x4x3xf32> {
    %c2 = arith.constant 2 : index
    %b = tensor.extract_slice %rhs[0, 0, 0] [%n, 8, %c2] [1, 1, 1]
      : tensor<?x8x?xf32> to tensor<?x8x?xf32>
    %r = "onnx.MatMul"(%lhs, %b)
      : (tensor<?x4x8xf32>, tensor<?x8x?xf32>) -> tensor<?x4x3xf32>
    return %r : tensor<?x4x3xf32>
  }
}

// -----

module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: "onnx.Gemm"
  // CHECK: return
  func.func @main_graph(%a: tensor<?x8xf32>,
                        %rhs: tensor<8x?xf32>) -> tensor<?x3xf32> {
    %c2 = arith.constant 2 : index
    %b = tensor.extract_slice %rhs[0, 0] [8, %c2] [1, 1]
      : tensor<8x?xf32> to tensor<8x?xf32>
    %r = "onnx.Gemm"(%a, %b)
      : (tensor<?x8xf32>, tensor<8x?xf32>) -> tensor<?x3xf32>
    return %r : tensor<?x3xf32>
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
