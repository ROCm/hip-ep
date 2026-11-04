// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.non_max_suppression lowers to llvm.call
// @wrap_non_max_suppression with f32 thresholds and dynamic extents.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @nms_static(
      %ctx: !hip.context,
      %boxes: memref<2x6x4xf32, 1>,
      %scores: memref<2x3x6xf32, 1>,
      %selected: memref<24x3xi64, 1>,
      %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nms_static
    // CHECK-DAG: llvm.mlir.constant(5.000000e-01 : f32) : f32
    // CHECK-DAG: llvm.mlir.constant(2.500000e-01 : f32) : f32
    hip.non_max_suppression(%ctx)
        ins(%boxes, %scores : memref<2x6x4xf32, 1>, memref<2x3x6xf32, 1>)
        outs(%selected, %count : memref<24x3xi64, 1>, memref<i32, 1>)
        {center_point_box = 1, has_score_threshold = 1,
         iou_threshold = 5.000000e-01 : f32, max_output_boxes_per_class = 4,
         score_threshold = 2.500000e-01 : f32}
    // CHECK: llvm.call @wrap_non_max_suppression
    return
  }

  func.func @nms_dynamic(
      %ctx: !hip.context,
      %boxes: memref<1x?x4xf32, 1>,
      %scores: memref<1x1x?xf32, 1>,
      %selected: memref<?x3xi64, 1>,
      %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nms_dynamic
    hip.non_max_suppression(%ctx)
        ins(%boxes, %scores : memref<1x?x4xf32, 1>, memref<1x1x?xf32, 1>)
        outs(%selected, %count : memref<?x3xi64, 1>, memref<i32, 1>)
        {center_point_box = 0, has_score_threshold = 0,
         iou_threshold = 0.699999988 : f32, max_output_boxes_per_class = 2000,
         score_threshold = 0.000000e+00 : f32}
    // CHECK: llvm.call @wrap_non_max_suppression
    return
  }
}
