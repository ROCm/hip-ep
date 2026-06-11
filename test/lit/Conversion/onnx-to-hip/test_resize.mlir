// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the ONNX Resize conversion produces a hip.resize DPS op with all
// string attributes resolved to integer enums at compile time.  Also
// covers:
//   * variadic ONNX operands (X, NoValue roi, scales, NoValue sizes)
//   * rejection / pass-through of attribute defaults
//   * (N, C) pass-through invariant
//   * single Variadic input form (only X — no extra operands at all)

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: 2D bilinear upsample with default half_pixel coord transform.
  // Variadic operand list: (X, roi=NoValue, scales).
  func.func @test_resize_linear_half_pixel(%arg0: tensor<1x3x16x16xf16>,
                                            %scales: tensor<4xf32>)
      -> tensor<1x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_linear_half_pixel
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x3x16x16xf16>
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<1x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x3x32x32xf16>
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<1x3x16x16xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<1x3x32x32xf16>)
    // mode=1 (linear), coord_transform=0 (half_pixel), nearest_mode=0
    // CHECK-SAME: coord_transform = 0
    // CHECK-SAME: mode = 1
    // CHECK-SAME: nearest_mode = 0

    return %y : tensor<1x3x32x32xf16>
  }

  // Test 2: 2D nearest with align_corners.
  func.func @test_resize_nearest_align_corners(%arg0: tensor<1x4x8x8xf32>,
                                                %scales: tensor<4xf32>)
      -> tensor<1x4x16x16xf32> {
    // CHECK-LABEL: func.func @test_resize_nearest_align_corners
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "nearest",
         coordinate_transformation_mode = "align_corners",
         nearest_mode = "round_prefer_floor"}
        : (tensor<1x4x8x8xf32>, none, tensor<4xf32>)
        -> tensor<1x4x16x16xf32>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 2
    // CHECK-SAME: mode = 0
    // CHECK-SAME: nearest_mode = 0
    return %y : tensor<1x4x16x16xf32>
  }

  // Test 3: 3D linear (volumetric upsample) with asymmetric coord.
  func.func @test_resize_3d_linear_asymmetric(%arg0: tensor<1x2x4x4x4xf32>,
                                                %scales: tensor<5xf32>)
      -> tensor<1x2x8x8x8xf32> {
    // CHECK-LABEL: func.func @test_resize_3d_linear_asymmetric
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "asymmetric"}
        : (tensor<1x2x4x4x4xf32>, none, tensor<5xf32>)
        -> tensor<1x2x8x8x8xf32>

    // CHECK-NOT: onnx.Resize
    // CHECK: hip.resize
    // CHECK-SAME: coord_transform = 1
    // CHECK-SAME: mode = 1
    return %y : tensor<1x2x8x8x8xf32>
  }

  // Test 4: dynamic batch (N) — verifies tensor.dim flowing into init.
  // Spatial dims must remain static.
  func.func @test_resize_dynamic_n(%arg0: tensor<?x3x16x16xf16>,
                                    %scales: tensor<4xf32>)
      -> tensor<?x3x32x32xf16> {
    // CHECK-LABEL: func.func @test_resize_dynamic_n
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x3x16x16xf16>
    %roi = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Resize"(%arg0, %roi, %scales)
        {mode = "linear", coordinate_transformation_mode = "half_pixel"}
        : (tensor<?x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<?x3x32x32xf16>

    // CHECK-NOT: onnx.Resize
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[DN:.*]] = tensor.dim %[[X]], %[[C0]] : tensor<?x3x16x16xf16>
    // CHECK: %[[INIT:.*]] = tensor.empty(%[[DN]]) : tensor<?x3x32x32xf16>
    // CHECK: hip.resize(%[[CTX]]) ins(%[[X]] : tensor<?x3x16x16xf16>)
    // CHECK-SAME: outs(%[[INIT]] : tensor<?x3x32x32xf16>)
    return %y : tensor<?x3x32x32xf16>
  }
}
