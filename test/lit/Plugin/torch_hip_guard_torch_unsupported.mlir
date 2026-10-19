// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: torch_hip_plugin
//
// torch_hip plugin guard: a module that contains `torch` dialect ops must fail
// with a clear [torch_hip] message while the conversion is not implemented.
// The Torch dialect is registered only from PR B on, so the op is written in
// generic form and parsed with --allow-unregistered-dialect.
//===----------------------------------------------------------------------===//

// Direct pass invocation:
// RUN: not hip-mlir-opt %s --allow-unregistered-dialect --convert-torch-to-hip 2>&1 | FileCheck %s
// Slot wiring: --hipdnn-pipeline inserts the pass at AfterOnnxLoopOutline
// (addPluginPassesForSlot), so the same error must appear from the pipeline.
// RUN: not hip-mlir-opt %s --allow-unregistered-dialect --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: [torch_hip] Torch input is not supported yet
func.func @main_graph(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "torch.aten.add.Tensor"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
