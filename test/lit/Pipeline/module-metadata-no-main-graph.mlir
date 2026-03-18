// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata without @main_graph.
//
// Verifies that the pass succeeds gracefully when @main_graph is absent —
// metadata generation is skipped (no error).  This allows single-function
// unit tests and partial-pipeline tests that do not carry @main_graph.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hip %s | FileCheck %s

// CHECK: func.func @other_func

module {
  func.func @other_func(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
