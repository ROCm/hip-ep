// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_static_plugins
//
// Pipeline-slot half of the plugin surface: the sample requests
// AfterConvertOnnxToHip, so --hipdnn-pipeline inserts the pass at that slot
// (addPluginPassesForSlot resolves it by name) and it fires on @main_graph.
// The companion test invokes the pass directly; this one proves the slot
// wiring. UNSUPPORTED unless the sample was selected (hip_static_plugins).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: [hip-ep-sample] visited main_graph
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32> {onnx.name = "input"})
      -> (tensor<2x3x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.Identity"(%arg0) {onnx_node_name = "identity_node"}
        : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    "onnx.Return"(%0) : (tensor<2x3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
