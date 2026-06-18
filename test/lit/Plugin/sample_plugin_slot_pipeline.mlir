// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_plugins_enabled
//
// Validates the pipeline-SLOT half of the plugin ABI end-to-end: the sample
// plugin's RegisterCallbacks calls
//   requestPipelineSlot(AfterConvertOnnxToHip, "hip-ep-sample-print-functions")
// so when the production --hipdnn-pipeline runs with the plugin loaded,
// lib/Dialect/Transforms/Pipelines.cpp::addPluginPassesForSlot resolves that
// pass by name and inserts it at the slot. The pass then fires on @main_graph,
// proving BOTH that the slot request was honored AND that the plugin-registered
// pass resolves in the host registry (via the HIPDNN_ENABLE_PLUGINS symbol
// export). The companion test sample_plugin_pass.mlir invokes the pass directly
// (--hip-ep-sample-print-functions); this one proves the slot wiring.
//===----------------------------------------------------------------------===//

// RUN: env HIP_EP_PLUGINS=%hip-ep-sample-plugin \
// RUN:   hip-mlir-opt %s --hipdnn-pipeline 2>&1 \
// RUN:   | FileCheck %s

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
