// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-fix-loop-accumulator-offset. The pass rewrites a
// frozen chunk-append `memref.subview` OFFSET (`memref.dim %v_in, %cN`) in an
// outlined hip.loop body to the real per-iter chunk start: seqlens_k[iter] via
// a synchronized hip.readback_scalar of the start gather, or iter*chunk_size.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-fix-loop-accumulator-offset --split-input-file %s 2>&1 | FileCheck %s

// CASE 1: canonical windowed-attention body (real bufferized shape). Two
// seqlens gathers read start = seqlens_k[iter] and end = seqlens_k[iter+1]; the
// pass must read back the FIRST (start) gather and use it as the chunk-append
// offset, while leaving the static-0 self-copy subview alone.
// CHECK-LABEL: func.func private @main_graph_loop_body_n0
// CHECK: %[[G0:.*]] = memref.alloc() {{.*}} : memref<1xi32>
// CHECK: hip.gather(%arg0) ins(%arg4, %{{.*}} : memref<?xi32>, memref<1xi64>) outs(%[[G0]] : memref<1xi32>)
// CHECK: %[[S:.*]] = hip.readback_scalar(%arg0, %[[G0]] : memref<1xi32>) -> i32
// CHECK: %[[SI:.*]] = arith.index_cast %[[S]] : i32 to index
// Self-copy subview keeps its static-0 offset (frozen dim is only in SIZES):
// CHECK: memref.subview %arg9[0, 0, 0]
// Chunk-append subview offset is now the readback start index, NOT memref.dim:
// CHECK: memref.subview %arg9[0, %[[SI]], 0] {{\[}}1, %{{.*}}, 1152]
module {
  memref.global "private" constant @hip_ext_constant_373 : memref<i64> = dense<1>
  func.func private @main_graph_loop_body_n0(%arg0: !hip.context, %arg1: memref<i64>, %arg2: memref<ui8>, %arg3: memref<1x?x1152xf16>, %arg4: memref<?xi32>, %arg5: memref<1x?x1152xf16>, %arg6: memref<1xi32>, %arg7: memref<1x?x1152xf16>, %arg8: memref<1x?x1152xf16>, %arg9: memref<1x?x1152xf16> {bufferize.result}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @hip_ext_constant_373 : memref<i64>
    %1 = hip.readback_scalar(%arg0, %arg1 : memref<i64>) -> i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
    memref.store %1, %alloc[%c0] : memref<1xi64>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xi32>
    hip.gather(%arg0) ins(%arg4, %alloc : memref<?xi32>, memref<1xi64>) outs(%alloc_0 : memref<1xi32>)
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
    hip.add(%arg0) ins(%alloc, %0 : memref<1xi64>, memref<i64>) outs(%alloc_1 : memref<1xi64>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xi32>
    hip.gather(%arg0) ins(%arg4, %alloc_1 : memref<?xi32>, memref<1xi64>) outs(%alloc_2 : memref<1xi32>)
    %dim_7 = memref.dim %arg8, %c1 : memref<1x?x1152xf16>
    %alloc_10 = memref.alloc(%dim_7) {alignment = 64 : i64} : memref<1x?x1152xf16>
    %dim_12 = memref.dim %arg3, %c1 : memref<1x?x1152xf16>
    %subview = memref.subview %arg9[0, 0, 0] [1, %dim_12, 1152] [1, 1, 1] : memref<1x?x1152xf16> to memref<1x?x1152xf16, strided<[?, 1152, 1]>>
    memref.copy %arg3, %subview : memref<1x?x1152xf16> to memref<1x?x1152xf16, strided<[?, 1152, 1]>>
    %dim_13 = memref.dim %arg3, %c1 : memref<1x?x1152xf16>
    %subview_14 = memref.subview %arg9[0, %dim_13, 0] [1, %dim_7, 1152] [1, 1, 1] : memref<1x?x1152xf16> to memref<1x?x1152xf16, strided<[?, 1152, 1], offset: ?>>
    memref.copy %alloc_10, %subview_14 : memref<1x?x1152xf16> to memref<1x?x1152xf16, strided<[?, 1152, 1], offset: ?>>
    return
  }
}

// -----

// CASE 2: fixed-stride fallback - a loop body with NO seqlens gather. The pass
// reads the iter scalar back synchronized and uses iter * chunk_size (the
// subview's static SIZE for that dim) as the chunk-append offset.
// CHECK-LABEL: func.func private @fixed_stride_loop_body_n0
// CHECK: %[[IT:.*]] = hip.readback_scalar(%arg0, %arg1 : memref<i64>) -> i64
// CHECK: %[[ITI:.*]] = arith.index_cast %[[IT]] : i64 to index
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[OFF:.*]] = arith.muli %[[ITI]], %[[C8]]
// CHECK: memref.subview %arg5[0, %[[OFF]], 0] {{\[}}1, 8, 1152]
module {
  func.func private @fixed_stride_loop_body_n0(%arg0: !hip.context, %arg1: memref<i64>, %arg2: memref<ui8>, %arg3: memref<1x?x1152xf16>, %arg4: memref<1x8x1152xf16>, %arg5: memref<1x?x1152xf16> {bufferize.result}) {
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg3, %c1 : memref<1x?x1152xf16>
    %subview = memref.subview %arg5[0, %dim, 0] [1, 8, 1152] [1, 1, 1] : memref<1x?x1152xf16> to memref<1x8x1152xf16, strided<[?, 1152, 1], offset: ?>>
    memref.copy %arg4, %subview : memref<1x8x1152xf16> to memref<1x8x1152xf16, strided<[?, 1152, 1], offset: ?>>
    return
  }
}

// -----

// CASE 3: negative - not an outlined loop body (no !hip.context arg0, no
// {bufferize.result}). The pass must leave the IR untouched.
// CHECK-LABEL: func.func @not_a_loop_body
// CHECK-NOT: hip.readback_scalar
// CHECK: %[[D:.*]] = memref.dim
// CHECK: memref.subview %arg1[0, %[[D]], 0]
func.func @not_a_loop_body(%arg0: memref<1x4x1152xf16>, %arg1: memref<1x?x1152xf16>) {
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg1, %c1 : memref<1x?x1152xf16>
  %subview = memref.subview %arg1[0, %dim, 0] [1, 4, 1152] [1, 1, 1] : memref<1x?x1152xf16> to memref<1x4x1152xf16, strided<[?, 1152, 1], offset: ?>>
  memref.copy %arg0, %subview : memref<1x4x1152xf16> to memref<1x4x1152xf16, strided<[?, 1152, 1], offset: ?>>
  return
}
