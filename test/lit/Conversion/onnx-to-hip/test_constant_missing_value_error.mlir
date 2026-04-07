// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Constant without value or location attribute produces an error.
//
// Verifies the updated error message that mentions the location attribute
// (zero-copy path) as an accepted alternative to a dense value attribute.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

// CHECK: error: unsupported onnx.Constant form (expected dense value attribute or location attribute)
// CHECK-NEXT: %0 = "onnx.Constant"() {value_string = "bad"} : () -> tensor<2xf32>
// CHECK-NEXT: ^

module {
  func.func @main_graph() -> tensor<2xf32> {
    %0 = "onnx.Constant"() {value_string = "bad"} : () -> tensor<2xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<2xf32>) -> ()
  }
}
