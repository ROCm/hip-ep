// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.where lowering to llvm.call @wrap_where using a rank-agnostic
// signature: each operand's real shape is emitted as a stack array (alloca)
// and passed together with its rank, so ONNX Where with arbitrary rank and
// multidirectional (NumPy-style) broadcasting is supported.
//
// This test validates:
// - hip.where -> llvm.call @wrap_where
// - wrap_where signature: (state, cond_ptr, x_ptr, y_ptr, out_ptr,
//                          cond_shape_ptr, cond_rank,
//                          x_shape_ptr,    x_rank,
//                          y_shape_ptr,    y_rank,
//                          out_shape_ptr,  out_rank,
//                          data_type) -> i32
// - Each shape pointer is an llvm.alloca of [rank x i64]
// - Static dims become llvm.mlir.constant; dynamic dims are read from the
//   memref descriptor
// - Ranks != 4 are accepted (no NCHW assumption)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Same-shape 2D f32
  // CHECK-LABEL: llvm.func @where_static_f32_test
  func.func @where_static_f32_test(
      %ctx: !hip.context,
      %cond: memref<2x4xi1, 1>,
      %x: memref<2x4xf32, 1>,
      %y: memref<2x4xf32, 1>,
      %out: memref<2x4xf32, 1>) {
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<2x4xi1, 1>, memref<2x4xf32, 1>, memref<2x4xf32, 1>)
                    outs(%out : memref<2x4xf32, 1>)

    // Each operand's rank-2 shape is stored into a stack array.
    // CHECK: llvm.alloca {{.*}} x !llvm.array<2 x i64>
    // Signature: state + 4 data ptrs + 4 (shape_ptr, rank) pairs + data_type.
    // CHECK: llvm.call @wrap_where({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 2: Multidirectional broadcasting f16
  // cond [1, 4], x [2, 1], y [2, 4], out [2, 4]
  // CHECK-LABEL: llvm.func @where_broadcast_f16_test
  func.func @where_broadcast_f16_test(
      %ctx: !hip.context,
      %cond: memref<1x4xi1, 1>,
      %x: memref<2x1xf16, 1>,
      %y: memref<2x4xf16, 1>,
      %out: memref<2x4xf16, 1>) {
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<1x4xi1, 1>, memref<2x1xf16, 1>, memref<2x4xf16, 1>)
                    outs(%out : memref<2x4xf16, 1>)

    // CHECK: llvm.call @wrap_where({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic shapes - dim sizes must be read from memref descriptor.
  // CHECK-LABEL: llvm.func @where_dynamic_test
  func.func @where_dynamic_test(
      %ctx: !hip.context,
      %cond: memref<?x?xi1, 1>,
      %x: memref<?x?xf32, 1>,
      %y: memref<?x?xf32, 1>,
      %out: memref<?x?xf32, 1>) {
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<?x?xi1, 1>, memref<?x?xf32, 1>, memref<?x?xf32, 1>)
                    outs(%out : memref<?x?xf32, 1>)

    // At least one dynamic dim is extracted from the descriptor via
    // llvm.extractvalue {{.*}}[3, ...] and then stored into the shape array.
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: llvm.call @wrap_where({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 4: i64 element type for X/Y, rank-2
  // CHECK-LABEL: llvm.func @where_i64_test
  func.func @where_i64_test(
      %ctx: !hip.context,
      %cond: memref<3x5xi1, 1>,
      %x: memref<3x5xi64, 1>,
      %y: memref<3x5xi64, 1>,
      %out: memref<3x5xi64, 1>) {
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<3x5xi1, 1>, memref<3x5xi64, 1>, memref<3x5xi64, 1>)
                    outs(%out : memref<3x5xi64, 1>)

    // CHECK: llvm.call @wrap_where({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

    return
  }

  // Test 5: Non-4D rank (rank-5) to confirm there's no NCHW assumption.
  // cond [2, 1, 1, 3, 1], x [1, 4, 1, 1, 5], y [2, 4, 3, 3, 5],
  // out [2, 4, 3, 3, 5] - all with multidirectional broadcasting.
  // CHECK-LABEL: llvm.func @where_rank5_test
  func.func @where_rank5_test(
      %ctx: !hip.context,
      %cond: memref<2x1x1x3x1xi1, 1>,
      %x: memref<1x4x1x1x5xf32, 1>,
      %y: memref<2x4x3x3x5xf32, 1>,
      %out: memref<2x4x3x3x5xf32, 1>) {
    hip.where(%ctx) ins(%cond, %x, %y :
                        memref<2x1x1x3x1xi1, 1>,
                        memref<1x4x1x1x5xf32, 1>,
                        memref<2x4x3x3x5xf32, 1>)
                    outs(%out : memref<2x4x3x3x5xf32, 1>)

    // Rank-5 shape arrays: alloca [5 x i64] for each of the four operands.
    // CHECK: llvm.alloca {{.*}} x !llvm.array<5 x i64>
    // Ranks are passed as i64 constants alongside each shape pointer.
    // CHECK: llvm.call @wrap_where({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32

    return
  }
}
