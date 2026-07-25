// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata without @main_graph.
//
// Verifies that the pass emits an error when @main_graph is absent, since
// our pipeline always expects it.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --convert-onnx-to-hip %s 2>&1 | FileCheck %s

// CHECK: error: expected @main_graph function for metadata generation

module {
  func.func @other_func(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
