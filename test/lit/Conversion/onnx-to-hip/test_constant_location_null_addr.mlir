// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx.Constant with location attribute but null address (offset=0).
//
// A zero address is invalid -- the ORT bridge should never produce one.
// Verify the pass rejects it with a clear diagnostic.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --hip-externalize-constants %s 2>&1 | FileCheck %s

// convert-onnx-to-hip lowers to a hip.constant carrier; hip-externalize-constants
// rejects the null mem-addr.
// CHECK: error: hip.constant mem-addr has null address

module {
  func.func @main_graph() -> tensor<2x4xf32> {
    %0 = "onnx.Constant"() {location = "*/_ORT_MEM_ADDR_/*", offset = 0 : i64, size = 32 : i64} : () -> tensor<2x4xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
