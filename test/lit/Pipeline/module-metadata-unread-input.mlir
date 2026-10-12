// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// An unread @main_graph argument is not a runtime input. inference_compute
// would otherwise call prepare_input past the tensors the EP passes.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// CHECK: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 1, 3, 112, 112>]
// CHECK: func.func @main_graph(%{{.*}}: !hip.context, %{{.*}}: tensor<1x3x112x112xf32>)

module {
  func.func @main_graph(%data: tensor<1x3x112x112xf32>, %reshape26: tensor<4xi64>) -> tensor<1x3x112x112xf32> {
    "onnx.Return"(%data) {onnx_node_name = "/Return"} : (tensor<1x3x112x112xf32>) -> ()
  }
}
