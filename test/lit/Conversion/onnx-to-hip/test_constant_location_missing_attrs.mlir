// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Constant with location attribute but missing offset/size.
//
// The zero-copy path expects both offset and size integer attributes
// alongside the location sentinel. Verify the diagnostic when they
// are absent.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

// CHECK: error: onnx.Constant with location attribute missing location/offset/size
// CHECK-NEXT: %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*"} : () -> tensor<2x4xf32>
// CHECK-NEXT: ^

module {
  func.func @main_graph() -> tensor<2x4xf32> {
    %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*"} : () -> tensor<2x4xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
