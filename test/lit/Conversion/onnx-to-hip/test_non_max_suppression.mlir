// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX NonMaxSuppression is lowered to hip.non_max_suppression with a
// worst-case capacity init, a device count, hip.readback_dim and a slice.
//
// Test cases:
// 1. MaskRCNN RPN shape: dynamic N, constant max / iou, no score threshold
// 2. Static shapes, score threshold, center_point_box = 1
// 3. Unranked boxes / scores are cast to rank 3
// 4. Non-constant max_output_boxes_per_class stays in the ONNX dialect
// 5. Only boxes and scores: max_output_boxes_per_class defaults to 0
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: MaskRCNN RPN ---
  func.func @nms_rpn(%boxes: tensor<1x?x4xf32>, %scores: tensor<1x1x?xf32>) -> tensor<?x3xi64> {
    %max = "onnx.Constant"() {value = dense<2000> : tensor<1xi64>} : () -> tensor<1xi64>
    %iou = "onnx.Constant"() {value = dense<0.699999988> : tensor<1xf32>} : () -> tensor<1xf32>
    %sel = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou) {center_point_box = 0 : si64} : (tensor<1x?x4xf32>, tensor<1x1x?xf32>, tensor<1xi64>, tensor<1xf32>) -> tensor<?x3xi64>
    return %sel : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_rpn
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[B:.*]]: tensor<1x?x4xf32>, %[[S:.*]]: tensor<1x1x?xf32>)
  // CHECK-NOT: onnx.NonMaxSuppression
  // CHECK: tensor.dim %[[B]], %{{.*}} : tensor<1x?x4xf32>
  // CHECK: arith.minsi
  // CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}) : tensor<?x3xi64>
  // CHECK: %[[CNT:.*]] = tensor.empty() : tensor<i32>
  // CHECK: %[[NMS:.*]]:2 = hip.non_max_suppression(%[[CTX]]) ins(%[[B]], %[[S]] : tensor<1x?x4xf32>, tensor<1x1x?xf32>) outs(%[[INIT]], %[[CNT]] : tensor<?x3xi64>, tensor<i32>)
  // CHECK-SAME: center_point_box = 0 : i64, has_score_threshold = 0 : i64, iou_threshold = {{.*}} : f32, max_output_boxes_per_class = 2000 : i64
  // CHECK: %[[N:.*]] = hip.readback_dim(%[[CTX]], %[[NMS]]#1 : tensor<i32>) -> index
  // CHECK: tensor.extract_slice %[[NMS]]#0[0, 0] [%[[N]], 3] [1, 1] : tensor<?x3xi64> to tensor<?x3xi64>

  // --- Case 2: static, score threshold, center boxes ---
  func.func @nms_static(%boxes: tensor<2x6x4xf32>, %scores: tensor<2x3x6xf32>) -> tensor<?x3xi64> {
    %max = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %iou = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
    %thr = "onnx.Constant"() {value = dense<2.500000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
    %sel = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou, %thr) {center_point_box = 1 : si64} : (tensor<2x6x4xf32>, tensor<2x3x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
    return %sel : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_static
  // CHECK-NOT: onnx.NonMaxSuppression
  // CHECK: %[[CAP:.*]] = arith.constant 24 : index
  // CHECK: tensor.empty(%[[CAP]]) : tensor<?x3xi64>
  // CHECK: hip.non_max_suppression
  // CHECK-SAME: center_point_box = 1 : i64, has_score_threshold = 1 : i64, iou_threshold = 5.000000e-01 : f32, max_output_boxes_per_class = 4 : i64, score_threshold = 2.500000e-01 : f32
  // CHECK: hip.readback_dim

  // --- Case 3: unranked boxes / scores ---
  func.func @nms_unranked(%boxes: tensor<*xf32>, %scores: tensor<*xf32>) -> tensor<?x3xi64> {
    %max = "onnx.Constant"() {value = dense<100> : tensor<1xi64>} : () -> tensor<1xi64>
    %iou = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
    %sel = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou) {center_point_box = 0 : si64} : (tensor<*xf32>, tensor<*xf32>, tensor<1xi64>, tensor<1xf32>) -> tensor<?x3xi64>
    return %sel : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_unranked
  // CHECK-NOT: onnx.NonMaxSuppression
  // CHECK-DAG: %[[BC:.*]] = tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?x?xf32>
  // CHECK-DAG: %[[SC:.*]] = tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?x?xf32>
  // CHECK: hip.non_max_suppression(%{{[^)]*}}) ins(%[[BC]], %[[SC]] :

  // --- Case 4: runtime max_output_boxes_per_class is declined ---
  func.func @nms_runtime_max(%boxes: tensor<1x8x4xf32>, %scores: tensor<1x1x8xf32>, %max: tensor<1xi64>) -> tensor<?x3xi64> {
    %sel = "onnx.NonMaxSuppression"(%boxes, %scores, %max) : (tensor<1x8x4xf32>, tensor<1x1x8xf32>, tensor<1xi64>) -> tensor<?x3xi64>
    return %sel : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_runtime_max
  // CHECK: onnx.NonMaxSuppression
  // CHECK-NOT: hip.non_max_suppression

  // --- Case 5: defaults ---
  func.func @nms_defaults(%boxes: tensor<1x8x4xf32>, %scores: tensor<1x2x8xf32>) -> tensor<?x3xi64> {
    %sel = "onnx.NonMaxSuppression"(%boxes, %scores) : (tensor<1x8x4xf32>, tensor<1x2x8xf32>) -> tensor<?x3xi64>
    return %sel : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_defaults
  // CHECK: %[[CAP0:.*]] = arith.constant 0 : index
  // CHECK: tensor.empty(%[[CAP0]]) : tensor<?x3xi64>
  // CHECK: hip.non_max_suppression
  // CHECK-SAME: has_score_threshold = 0 : i64, iou_threshold = 0.000000e+00 : f32, max_output_boxes_per_class = 0 : i64
}
