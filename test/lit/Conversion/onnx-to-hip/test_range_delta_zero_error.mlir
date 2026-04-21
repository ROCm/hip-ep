// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Range with delta=0 should fail conversion.
//===----------------------------------------------------------------------===//
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

// CHECK: error: onnx.Range requires non-zero delta

module {
  func.func @main_graph() -> tensor<1xi64> {
    %start = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %limit = "onnx.Constant"() {value = dense<10> : tensor<i64>} : () -> tensor<i64>
    %delta = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1xi64>
    "onnx.Return"(%0) : (tensor<1xi64>) -> ()
  }
}
