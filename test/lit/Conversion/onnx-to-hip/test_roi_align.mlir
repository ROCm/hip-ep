// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX RoiAlign is lowered to hip.roi_align.
//
// Test cases:
// 1. Dynamic num_rois, default coordinate_transformation_mode (MaskRCNN-style)
// 2. Static shapes, max mode, output_half_pixel
// 3. Unranked rois / batch_indices are cast to the spec ranks
// 3b. All-unranked op is declined by UnrankedOpGuard
// 4. Dynamic channel count comes from X
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: dynamic num_rois, defaults ---
  func.func @roi_align_dynamic(%X: tensor<1x256x128x128xf32>, %R: tensor<?x4xf32>, %I: tensor<?xi64>) -> tensor<?x256x7x7xf32> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {mode = "avg", output_height = 7 : si64, output_width = 7 : si64, sampling_ratio = 2 : si64, spatial_scale = 2.500000e-01 : f32} : (tensor<1x256x128x128xf32>, tensor<?x4xf32>, tensor<?xi64>) -> tensor<?x256x7x7xf32>
    return %Y : tensor<?x256x7x7xf32>
  }

  // CHECK-LABEL: func.func @roi_align_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x256x128x128xf32>, %[[R:.*]]: tensor<?x4xf32>, %[[I:.*]]: tensor<?xi64>)
  // CHECK-NOT: onnx.RoiAlign
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[N:.*]] = tensor.dim %[[R]], %[[C0]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[N]]) : tensor<?x256x7x7xf32>
  // CHECK: hip.roi_align(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[R]], %[[I]] :
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x256x7x7xf32>)
  // CHECK-SAME: half_pixel = 1 : i64, mode = 0 : i64, output_height = 7 : i64, output_width = 7 : i64, sampling_ratio = 2 : i64, spatial_scale = 2.500000e-01 : f32

  // --- Case 2: static, max, output_half_pixel ---
  func.func @roi_align_max(%X: tensor<2x3x10x10xf16>, %R: tensor<5x4xf16>, %I: tensor<5xi64>) -> tensor<5x3x2x3xf16> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {coordinate_transformation_mode = "output_half_pixel", mode = "max", output_height = 2 : si64, output_width = 3 : si64, sampling_ratio = 0 : si64, spatial_scale = 1.000000e+00 : f32} : (tensor<2x3x10x10xf16>, tensor<5x4xf16>, tensor<5xi64>) -> tensor<5x3x2x3xf16>
    return %Y : tensor<5x3x2x3xf16>
  }

  // CHECK-LABEL: func.func @roi_align_max
  // CHECK-NOT: onnx.RoiAlign
  // CHECK: tensor.empty() : tensor<5x3x2x3xf16>
  // CHECK: hip.roi_align(%{{[^)]*}})
  // CHECK-SAME: half_pixel = 0 : i64, mode = 1 : i64, output_height = 2 : i64, output_width = 3 : i64, sampling_ratio = 0 : i64

  // --- Case 3: unranked rois / batch_indices, ranked result ---
  func.func @roi_align_unranked(%X: tensor<1x8x16x16xf32>, %R: tensor<*xf32>, %I: tensor<*xi64>) -> tensor<?x8x14x14xf32> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {output_height = 14 : si64, output_width = 14 : si64, sampling_ratio = 2 : si64, spatial_scale = 6.250000e-02 : f32} : (tensor<1x8x16x16xf32>, tensor<*xf32>, tensor<*xi64>) -> tensor<?x8x14x14xf32>
    return %Y : tensor<?x8x14x14xf32>
  }

  // CHECK-LABEL: func.func @roi_align_unranked
  // CHECK-NOT: onnx.RoiAlign
  // CHECK-DAG: %[[RC:.*]] = tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?xf32>
  // CHECK-DAG: %[[IC:.*]] = tensor.cast %{{.*}} : tensor<*xi64> to tensor<?xi64>
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}) : tensor<?x8x14x14xf32>
  // CHECK: hip.roi_align(%{{[^)]*}})
  // CHECK-SAME: ins(%{{.*}}, %[[RC]], %[[IC]] :

  // --- Case 3b: operands and result all unranked are declined by the
  // UnrankedOpGuard in convert-onnx-to-hip and stay in the ONNX dialect ---
  func.func @roi_align_all_unranked(%X: tensor<1x8x16x16xf32>, %R: tensor<*xf32>, %I: tensor<*xi64>) -> tensor<*xf32> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {output_height = 14 : si64, output_width = 14 : si64} : (tensor<1x8x16x16xf32>, tensor<*xf32>, tensor<*xi64>) -> tensor<*xf32>
    return %Y : tensor<*xf32>
  }

  // CHECK-LABEL: func.func @roi_align_all_unranked
  // CHECK: onnx.RoiAlign
  // CHECK-NOT: hip.roi_align

  // --- Case 4: dynamic channels ---
  func.func @roi_align_dyn_channels(%X: tensor<?x?x?x?xf32>, %R: tensor<?x4xf32>, %I: tensor<?xi64>) -> tensor<?x?x7x7xf32> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {output_height = 7 : si64, output_width = 7 : si64} : (tensor<?x?x?x?xf32>, tensor<?x4xf32>, tensor<?xi64>) -> tensor<?x?x7x7xf32>
    return %Y : tensor<?x?x7x7xf32>
  }

  // CHECK-LABEL: func.func @roi_align_dyn_channels
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x?x?xf32>, %[[R:.*]]: tensor<?x4xf32>, %[[I:.*]]: tensor<?xi64>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[N:.*]] = tensor.dim %[[R]], %[[C0]]
  // CHECK-DAG: %[[C:.*]] = tensor.dim %[[X]], %[[C1]]
  // CHECK: tensor.empty(%[[N]], %[[C]]) : tensor<?x?x7x7xf32>
  // CHECK: hip.roi_align(%[[CTX]])
  // CHECK-SAME: half_pixel = 1 : i64, mode = 0 : i64

  // rois must have 4 columns. The rejection comes before any operand is cast,
  // so the unranked X is left without a stray tensor.cast.
  func.func @roi_align_bad_rois_unranked_x(%X: tensor<*xf32>, %R: tensor<?x5xf32>, %I: tensor<?xi64>) -> tensor<?x8x7x7xf32> {
    %Y = "onnx.RoiAlign"(%X, %R, %I) {output_height = 7 : si64, output_width = 7 : si64} : (tensor<*xf32>, tensor<?x5xf32>, tensor<?xi64>) -> tensor<?x8x7x7xf32>
    return %Y : tensor<?x8x7x7xf32>
  }

  // CHECK-LABEL: func.func @roi_align_bad_rois_unranked_x
  // CHECK-NOT: tensor.cast
  // CHECK: onnx.RoiAlign
  // CHECK-NOT: hip.roi_align
  // CHECK: return
}
