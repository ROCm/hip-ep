// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: module metadata with dynamic shapes.
//
// Verifies that dynamic dimensions (?) are preserved as -1 in the metadata
// shape arrays, matching ShapedType::kDynamic.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 4>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -9223372036854775808, 3, -9223372036854775808>]
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 4>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: -9223372036854775808, 3, -9223372036854775808>]

module {
  func.func @main_graph(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
    "onnx.Return"(%arg0) {onnx_node_name = "/Return"} : (tensor<?x3x?xf32>) -> ()
  }
}
