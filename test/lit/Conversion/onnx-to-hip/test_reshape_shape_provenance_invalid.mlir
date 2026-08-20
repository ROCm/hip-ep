// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --split-input-file --verify-diagnostics

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>,
      %shape: tensor<3xi64>) -> tensor<?x?x?xf16> {
    // expected-error@+1 {{nonnegative shape proof requires host shape proof}}
    %result = "onnx.Reshape"(%data, %shape) {
      allowzero = 0 : si64, hip.host_shape_no_minus_one
    } : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?xf16>,
      %shape: tensor<1xi64>) {
    // expected-error@+1 {{onnx.Reshape expects exactly two operands and one result}}
    "onnx.Reshape"(%data, %shape)
      : (tensor<?xf16>, tensor<1xi64>) -> ()
    return
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %data, %c0 : tensor<?x?xf16>
    %d0i64 = arith.index_cast %d0 : index to i64
    %c1 = arith.constant 1 : i64
    %target = tensor.from_elements %d0i64, %c1, %c1 : tensor<3xi64>
    // expected-error@+1 {{host shape proof requires an input-dimension map}}
    %result = "onnx.Reshape"(%data, %target) {
      allowzero = 0 : si64, hip.host_shape_operand
    } : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?x?xf16>) -> tensor<?x?x3xf16> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %data, %c0 : tensor<?x?x?xf16>
    %d0i64 = arith.index_cast %d0 : index to i64
    %minusOne = arith.constant -1 : i64
    %three = arith.constant 3 : i64
    %target = tensor.from_elements %d0i64, %minusOne, %three : tensor<3xi64>
    // expected-error@+1 {{allowzero=1 with -1 requires positive proven target dimensions}}
    %result = "onnx.Reshape"(%data, %target) {
      allowzero = 1 : si64,
      hip.host_shape_input_dim_map = array<i64: -1, -1, -1>,
      hip.host_shape_operand
    } : (tensor<?x?x?xf16>, tensor<3xi64>) -> tensor<?x?x3xf16>
    return %result : tensor<?x?x3xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>,
      %other: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %otherD0 = tensor.dim %other, %c0 : tensor<?x?xf16>
    %otherD0i64 = arith.index_cast %otherD0 : index to i64
    %c1 = arith.constant 1 : i64
    %target = tensor.from_elements %otherD0i64, %c1, %c1 : tensor<3xi64>
    // expected-error@+1 {{invalid input-dimension equivalence proof}}
    %result = "onnx.Reshape"(%data, %target) {
      allowzero = 0 : si64,
      hip.host_shape_input_dim_map = array<i64: 0, -1, -1>,
      hip.host_shape_operand,
      hip.host_shape_no_minus_one
    } : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %target = "onnx.Constant"() {
      value = dense<[1, 1, 0]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    // expected-error@+1 {{out-of-rank Reshape target entry cannot be zero}}
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>,
      %shape: tensor<2xi64>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %unsafe = tensor.extract %shape[%c0] : tensor<2xi64>
    %c1 = arith.constant 1 : i64
    %target = tensor.from_elements %unsafe, %c1, %c1 : tensor<3xi64>
    // expected-error@+1 {{proven host shape contains unsafe scalar}}
    %result = "onnx.Reshape"(%data, %target) {
      allowzero = 0 : si64,
      hip.host_shape_input_dim_map = array<i64: -1, -1, -1>,
      hip.host_shape_operand,
      hip.host_shape_no_minus_one
    } : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %target = "onnx.Constant"() {
      value = dense<[-2, 1, 1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    // expected-error@+1 {{Reshape target dimensions must be at least -1}}
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %target = "onnx.Constant"() {
      value = dense<[-1, -1, 1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    // expected-error@+1 {{Reshape target may contain at most one -1}}
    %result = "onnx.Reshape"(%data, %target) {allowzero = 0 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %data: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
    %target = "onnx.Constant"() {
      value = dense<[0, -1, 1]> : tensor<3xi64>
    } : () -> tensor<3xi64>
    // expected-error@+1 {{Reshape cannot combine allowzero=1, zero, and -1}}
    %result = "onnx.Reshape"(%data, %target) {allowzero = 1 : si64}
      : (tensor<?x?xf16>, tensor<3xi64>) -> tensor<?x?x?xf16>
    return %result : tensor<?x?x?xf16>
  }
}
