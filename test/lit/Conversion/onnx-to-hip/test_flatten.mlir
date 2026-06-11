// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the ONNX Flatten lowering: every onnx.Flatten is decomposed into
// pure shape metadata ops (tensor.collapse_shape and/or tensor.expand_shape).
// No `hip.*` op or runtime kernel is produced.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: default axis = 1, rank-4 static input. Pure collapse, two groups
  //         [[0], [1, 2, 3]] -> tensor<2x60xf32>.
  func.func @test_flatten_axis1_static(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x60xf32> {
    // CHECK-LABEL: func.func @test_flatten_axis1_static
    %r = "onnx.Flatten"(%arg0) {axis = 1 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<2x60xf32>

    // CHECK-NOT: onnx.Flatten
    // CHECK-NOT: hip.flatten
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0{{\]}}, {{\[}}1, 2, 3{{\]\]}}
    // CHECK-SAME: tensor<2x3x4x5xf32> into tensor<2x60xf32>

    return %r : tensor<2x60xf32>
  }

  // Test 2: axis = 2 on a rank-4 static input. Reassoc [[0,1], [2,3]] -> 6x20.
  func.func @test_flatten_axis2_static(%arg0: tensor<2x3x4x5xf32>) -> tensor<6x20xf32> {
    // CHECK-LABEL: func.func @test_flatten_axis2_static
    %r = "onnx.Flatten"(%arg0) {axis = 2 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<6x20xf32>

    // CHECK-NOT: onnx.Flatten
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1{{\]}}, {{\[}}2, 3{{\]\]}}
    // CHECK-SAME: tensor<2x3x4x5xf32> into tensor<6x20xf32>

    return %r : tensor<6x20xf32>
  }

  // Test 3: axis = 0 corner case on a rank-3 static input. Requires
  //         collapse-then-expand: tensor<2x3x4> -> tensor<24> -> tensor<1x24>.
  func.func @test_flatten_axis0_static(%arg0: tensor<2x3x4xf32>) -> tensor<1x24xf32> {
    // CHECK-LABEL: func.func @test_flatten_axis0_static
    %r = "onnx.Flatten"(%arg0) {axis = 0 : si64}
        : (tensor<2x3x4xf32>) -> tensor<1x24xf32>

    // CHECK-NOT: onnx.Flatten
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1, 2{{\]\]}}
    // CHECK-SAME: tensor<2x3x4xf32> into tensor<24xf32>
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1{{\]\]}}
    // CHECK-SAME: tensor<24xf32> into tensor<1x24xf32>

    return %r : tensor<1x24xf32>
  }

  // Test 4: axis = r corner case (axis = 3 on rank-3 input). Trailing 1.
  //         tensor<2x3x4> -> tensor<24> -> tensor<24x1>.
  func.func @test_flatten_axis_eq_rank_static(%arg0: tensor<2x3x4xf32>) -> tensor<24x1xf32> {
    // CHECK-LABEL: func.func @test_flatten_axis_eq_rank_static
    %r = "onnx.Flatten"(%arg0) {axis = 3 : si64}
        : (tensor<2x3x4xf32>) -> tensor<24x1xf32>

    // CHECK-NOT: onnx.Flatten
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1, 2{{\]\]}}
    // CHECK-SAME: tensor<2x3x4xf32> into tensor<24xf32>
    // CHECK: tensor.expand_shape %{{.*}} {{\[\[}}0, 1{{\]\]}}
    // CHECK-SAME: tensor<24xf32> into tensor<24x1xf32>

    return %r : tensor<24x1xf32>
  }

  // Test 5: dynamic batch dim with default axis = 1. The collapsed inner
  //         group is fully static (3*4*5 = 60) so the output has shape
  //         <?x60xf16>. tensor.collapse_shape naturally propagates the
  //         dynamic outer dim.
  func.func @test_flatten_dynamic_batch(%arg0: tensor<?x3x4x5xf16>) -> tensor<?x60xf16> {
    // CHECK-LABEL: func.func @test_flatten_dynamic_batch
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x3x4x5xf16>)
    %r = "onnx.Flatten"(%arg0) {axis = 1 : si64}
        : (tensor<?x3x4x5xf16>) -> tensor<?x60xf16>

    // CHECK-NOT: onnx.Flatten
    // CHECK: tensor.collapse_shape %[[ARG]] {{\[\[}}0{{\]}}, {{\[}}1, 2, 3{{\]\]}}
    // CHECK-SAME: tensor<?x3x4x5xf16> into tensor<?x60xf16>

    return %r : tensor<?x60xf16>
  }

  // Test 6: negative axis = -1 normalises to r-1 = 3 on rank-4. Single
  //         collapse with reassoc [[0,1,2], [3]] -> 24x5.
  func.func @test_flatten_negative_axis(%arg0: tensor<2x3x4x5xf32>) -> tensor<24x5xf32> {
    // CHECK-LABEL: func.func @test_flatten_negative_axis
    %r = "onnx.Flatten"(%arg0) {axis = -1 : si64}
        : (tensor<2x3x4x5xf32>) -> tensor<24x5xf32>

    // CHECK-NOT: onnx.Flatten
    // CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1, 2{{\]}}, {{\[}}3{{\]\]}}
    // CHECK-SAME: tensor<2x3x4x5xf32> into tensor<24x5xf32>

    return %r : tensor<24x5xf32>
  }
}
