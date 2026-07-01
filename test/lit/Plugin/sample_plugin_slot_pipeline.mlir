// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_static_plugins
//
// Validates the pipeline-SLOT half of the plugin surface end-to-end: the sample
// plugin's hipEpRegisterPlugin_sample calls
//   requestPipelineSlot(AfterConvertOnnxToHip, "hip-ep-sample-print-functions")
// so when the production --hipdnn-pipeline runs, lib/Dialect/Transforms/
// Pipelines.cpp::addPluginPassesForSlot resolves that pass by name and inserts
// it at the slot. The pass then fires on @main_graph, proving BOTH that the slot
// request was honored AND that the plugin-registered pass resolves in the host
// registry (the plugin is statically linked, so it shares the host's single
// registry). The companion test sample_plugin_pass.mlir invokes the pass
// directly (--hip-ep-sample-print-functions); this one proves the slot wiring.
//
// hip_static_plugins is set when the build selected the sample plugin
// (-DHIPDNN_EP_COMPILER_PLUGINS=sample). The default build selects no plugins,
// so this test is UNSUPPORTED there rather than failing.
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
