// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// An unranked Clip bound is unsupported by both PackBroadcastTo4D and the HIP
// broadcast destination builder. Both patterns must leave the ONNX operation
// unchanged, without packing the data or constructing an invalid HIP op.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @clip_5d_unranked_bound
  // CHECK-SAME: %[[X:[^:]+]]: tensor<2x3x4x5x6xf32>, %[[LO:[^:]+]]: tensor<*xf32>
  // CHECK-NEXT: %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: %[[RESULT:.*]] = "onnx.Clip"(%[[X]], %[[LO]], %[[NONE]]) : (tensor<2x3x4x5x6xf32>, tensor<*xf32>, none) -> tensor<2x3x4x5x6xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2x3x4x5x6xf32>
  func.func @clip_5d_unranked_bound(%x: tensor<2x3x4x5x6xf32>, %lo: tensor<*xf32>)
      -> tensor<2x3x4x5x6xf32> {
    %n = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Clip"(%x, %lo, %n) : (tensor<2x3x4x5x6xf32>, tensor<*xf32>, none) -> tensor<2x3x4x5x6xf32>
    return %y : tensor<2x3x4x5x6xf32>
  }
}
