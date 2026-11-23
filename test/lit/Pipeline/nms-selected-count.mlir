// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Regression guard for a data-dependent NonMaxSuppression graph output.
//
// The kernel writes into a B*C*S capacity buffer and reports the selected-row
// count separately. Returning a view of that buffer makes hip.alloc_output
// request the capacity (for example [6, 3] when one row was selected). The
// output must be a distinct buffer sized by hip.readback_dim.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK:         hip.non_max_suppression
// CHECK:         %[[N:.*]] = hip.readback_dim
// CHECK:         hip.alloc_output(%{{.*}}, %[[N]]) {out_idx = 0 : i64} : memref<?x3xi64>

module {
  func.func @main_graph(%boxes: tensor<1x6x4xf32> {onnx.name = "boxes"},
                        %scores: tensor<1x1x6xf32> {onnx.name = "scores"},
                        %max: tensor<i64> {onnx.name = "max"},
                        %iou: tensor<f32> {onnx.name = "iou"},
                        %score: tensor<f32> {onnx.name = "score"})
      -> (tensor<?x3xi64> {onnx.name = "selected"}) {
    %0 = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou, %score) {
      center_point_box = 0 : si64
    } : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<i64>, tensor<f32>, tensor<f32>) -> tensor<?x3xi64>
    return %0 : tensor<?x3xi64>
  }
}
