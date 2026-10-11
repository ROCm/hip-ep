// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// GatherND's runtime ABI reads indices as int64 and carries no width. Reject
// i32 before the conversion emits shape arithmetic or a destination.
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(
      %data: tensor<2x2xf32>,
      %indices: tensor<2x2xi32>) -> tensor<2xf32> {
    // CHECK: error: GatherND indices element type must be i64
    %result = "onnx.GatherND"(%data, %indices)
        : (tensor<2x2xf32>, tensor<2x2xi32>) -> tensor<2xf32>
    return %result : tensor<2xf32>
  }
}

// CHECK-NOT: tensor.dim
// CHECK-NOT: tensor.empty
// CHECK-NOT: hip.gather_nd
