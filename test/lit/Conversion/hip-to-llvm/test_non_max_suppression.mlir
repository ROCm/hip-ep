// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.non_max_suppression lowers to one call into wrap_non_max_suppression:
//   (state, boxes, scores, max, iou, score, selected, count,
//    num_batches, num_classes, num_boxes, capacity, center_point_box,
//    has_score_threshold)

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @nms(%ctx: !hip.context, %boxes: memref<1x?x4xf32, 1>,
                 %scores: memref<1x1x?xf32, 1>, %max: memref<i64, 1>,
                 %iou: memref<f32, 1>, %score: memref<f32, 1>,
                 %selected: memref<?x3xi64, 1>, %count: memref<i32, 1>) {
    // CHECK-LABEL: llvm.func @nms
    // CHECK: llvm.call @wrap_non_max_suppression({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32
    hip.non_max_suppression(%ctx)
        ins(%boxes, %scores, %max, %iou, %score :
            memref<1x?x4xf32, 1>, memref<1x1x?xf32, 1>, memref<i64, 1>,
            memref<f32, 1>, memref<f32, 1>)
        outs(%selected, %count : memref<?x3xi64, 1>, memref<i32, 1>)
        {center_point_box = 0 : i64, has_score_threshold = true}
    return
  }
}
