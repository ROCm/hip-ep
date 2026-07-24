// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// When a graph output is returned through memref.collapse_shape (internal
// compute rank differs from ONNX / func.return rank), hipdnn_ep_alloc_output
// must receive the RETURNED rank and dims — not the root alloc rank.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module attributes {
  hipdnn.input_count = 1 : i64,
  hipdnn.input_shapes = [array<i64: -9223372036854775808, 3, 16>],
  hipdnn.input_element_sizes = array<i64: 2>,
  hipdnn.output_count = 1 : i64,
  hipdnn.output_shapes = [array<i64: -9223372036854775808, 2560>],
  hipdnn.output_element_sizes = array<i64: 2>
} {
  // Gemma3-class vision encoder: internal BSH, ONNX/OGA image_features is SH.
  // CHECK-LABEL: llvm.func @vision_internal3_return2
  // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64>
  // CHECK:       llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
  // CHECK:       llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
  // CHECK:       %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK:       llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %{{.*}}, %[[SHAPE]], %[[RANK]], %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 0]
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 1]
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 2]
  func.func @vision_internal3_return2(%ctx: !hip.context, %seq: index)
      -> memref<?x2560xf16> {
    %out = hip.alloc_output(%ctx, %seq) {out_idx = 0 : i64} : memref<1x?x2560xf16>
    %ret = memref.collapse_shape %out [[0, 1], [2]]
        : memref<1x?x2560xf16> into memref<?x2560xf16>
    return %ret : memref<?x2560xf16>
  }

  // Conv-style flatten: internal rank 4, returned rank 2 (both static).
  // CHECK-LABEL: llvm.func @collapse_static
  // CHECK:       %[[SHAPE2:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64>
  // CHECK:       llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
  // CHECK:       llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr
  // CHECK:       %[[RANK2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK:       llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %{{.*}}, %[[SHAPE2]], %[[RANK2]], %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 0]
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 1]
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 2]
  // CHECK:       llvm.insertvalue {{.*}}, {{.*}}[3, 3]
  func.func @collapse_static(%ctx: !hip.context) -> memref<1x200704xf32> {
    %out = hip.alloc_output(%ctx) {out_idx = 0 : i64} : memref<1x64x56x56xf32>
    %ret = memref.collapse_shape %out [[0], [1, 2, 3]]
        : memref<1x64x56x56xf32> into memref<1x200704xf32>
    return %ret : memref<1x200704xf32>
  }
}
