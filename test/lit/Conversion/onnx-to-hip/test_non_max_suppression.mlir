// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // All five inputs present, dynamic box count (SSD postprocessor shape).
  // The capacity is B * C * S rows; the readback count slices it.
  func.func @nms_all_inputs(%boxes: tensor<1x?x4xf32>, %scores: tensor<1x1x?xf32>,
                            %max: tensor<i64>, %iou: tensor<f32>,
                            %score: tensor<f32>) -> tensor<?x3xi64> {
    %y = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou, %score) {center_point_box = 0 : si64}
        : (tensor<1x?x4xf32>, tensor<1x1x?xf32>, tensor<i64>, tensor<f32>, tensor<f32>) -> tensor<?x3xi64>
    return %y : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_all_inputs
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[B:.*]]: tensor<1x?x4xf32>, %[[S:.*]]: tensor<1x1x?xf32>, %[[M:.*]]: tensor<i64>, %[[I:.*]]: tensor<f32>, %[[T:.*]]: tensor<f32>)
  // CHECK: %[[DIM:.*]] = tensor.dim %[[B]], %{{.*}} : tensor<1x?x4xf32>
  // CHECK: %[[CAP:.*]] = arith.muli %{{.*}}, %[[DIM]] : index
  // CHECK: %[[SEL:.*]] = tensor.empty(%[[CAP]]) : tensor<?x3xi64>
  // CHECK: %[[CNT:.*]] = tensor.empty() : tensor<i32>
  // CHECK: %[[NMS:.*]]:2 = hip.non_max_suppression(%[[CTX]]) ins(%[[B]], %[[S]], %[[M]], %[[I]], %[[T]] : tensor<1x?x4xf32>, tensor<1x1x?xf32>, tensor<i64>, tensor<f32>, tensor<f32>) outs(%[[SEL]], %[[CNT]] : tensor<?x3xi64>, tensor<i32>) {center_point_box = 0 : i64, has_score_threshold = true}
  // CHECK: %[[N:.*]] = hip.readback_dim(%[[CTX]], %[[NMS]]#1 : tensor<i32>) -> index
  // CHECK: tensor.extract_slice %[[NMS]]#0[0, 0] [%[[N]], 3] [1, 1] : tensor<?x3xi64> to tensor<?x3xi64>

  // Only boxes and scores: absent inputs become constants, and score
  // filtering is disabled by attribute.
  func.func @nms_defaults(%boxes: tensor<2x6x4xf32>, %scores: tensor<2x3x6xf32>) -> tensor<?x3xi64> {
    %y = "onnx.NonMaxSuppression"(%boxes, %scores) {center_point_box = 1 : si64}
        : (tensor<2x6x4xf32>, tensor<2x3x6xf32>) -> tensor<?x3xi64>
    return %y : tensor<?x3xi64>
  }

  // CHECK-LABEL: func.func @nms_defaults
  // CHECK-DAG: %[[MAX:.*]] = hip.constant {{.*}}value = dense<0> : tensor<i64>{{.*}} : tensor<i64>
  // CHECK-DAG: hip.constant {{.*}}value = dense<0.000000e+00> : tensor<f32>{{.*}} : tensor<f32>
  // CHECK: tensor.empty(%{{.*}}) : tensor<?x3xi64>
  // CHECK: hip.non_max_suppression(%{{.*}}) ins(%{{.*}}, %{{.*}}, %[[MAX]], %{{.*}}, %{{.*}} : tensor<2x6x4xf32>, tensor<2x3x6xf32>, tensor<i64>, tensor<f32>, tensor<f32>)
  // CHECK-SAME: {center_point_box = 1 : i64, has_score_threshold = false}
}
