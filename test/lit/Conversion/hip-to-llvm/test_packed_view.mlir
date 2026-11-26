// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// The view's aligned pointer is the source's aligned pointer advanced by the
// source offset plus element_offset, and its own offset is zero, so a
// lowering that loads only the aligned pointer addresses the span.

// CHECK-LABEL: llvm.func @packed_view
// CHECK: %[[ALIGNED:.*]] = llvm.extractvalue %{{.*}}[1]
// CHECK: %[[BASE_OFF:.*]] = llvm.extractvalue %{{.*}}[2]
// CHECK: %[[SHIFT:.*]] = llvm.add %[[BASE_OFF]], %{{.*}}
// CHECK: %[[PTR:.*]] = llvm.getelementptr %[[ALIGNED]][%[[SHIFT]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.insertvalue %[[PTR]], %{{.*}}[1]
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.insertvalue %[[ZERO]], %{{.*}}[2]
func.func @packed_view(%cat: memref<1x8x2x2xf32>) -> memref<1x4x2x2xf32> {
  %c16 = arith.constant 16 : index
  %slot = hip.packed_view %cat[%c16] : memref<1x8x2x2xf32> to memref<1x4x2x2xf32>
  return %slot : memref<1x4x2x2xf32>
}
