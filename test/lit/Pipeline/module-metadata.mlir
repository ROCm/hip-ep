// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata generation in convert-onnx-to-hip.
//
// Verifies that generateModuleMetadata() sets the six hipdnn.* module
// attributes from @main_graph's original tensor signature before lowering.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 2, 4>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 1, 3, 16>, array<i64: 1, 3, 16>]
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 2>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 1, 3, 16>]

module {
  func.func @main_graph(%arg0: tensor<1x3x16xf16>, %arg1: tensor<1x3x16xf32>) -> tensor<1x3x16xf16> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"} : (tensor<1x3x16xf16>) -> ()
  }
}
