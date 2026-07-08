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

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-externalize-constants %s 2>&1 | FileCheck %s

// convert-onnx-to-hip lowers the malformed constant to a hip.constant carrier;
// hip-externalize-constants emits the diagnostic when offset/size are absent.
// CHECK: error: hip.constant with location missing location/offset/size

module {
  func.func @main_graph() -> tensor<2x4xf32> {
    %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*"} : () -> tensor<2x4xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
