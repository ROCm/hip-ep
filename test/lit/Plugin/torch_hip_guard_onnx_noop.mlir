// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: torch_hip_plugin
//
// torch_hip plugin guard: convert-torch-to-hip is linked into every host binary
// of a plugin-enabled build, so it must leave non-Torch (ONNX) modules
// untouched. UNSUPPORTED unless `torch_hip` is in HIPDNN_EP_COMPILER_PLUGINS.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --allow-unregistered-dialect --convert-torch-to-hip | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK: "onnx.Add"
func.func @main_graph(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
