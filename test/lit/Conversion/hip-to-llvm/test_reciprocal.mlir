// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP reciprocal operation is correctly lowered to LLVM call
// to unified wrap_power runtime function.
//
// This test validates:
// - hip.reciprocal → llvm.call @wrap_power(..., alpha=0, beta=1, gamma=-1)
// - num_elements computation for static and dynamic shapes
// - Data type enum passed as i64 (f32=0, f16=1, bf16=2)
// - Eight LLVM operands: state, input, output, num_elements, data_type,
//   alpha, beta, gamma (all f64 for the three power coefficients)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 1D f32 - num_elements = 128, data_type = 0 (f32)
  func.func @reciprocal_static_1d_f32(
      %ctx: !hip.context,
      %input: memref<128xf32, 1>,
      %output: memref<128xf32, 1>) {
    // CHECK-LABEL: llvm.func @reciprocal_static_1d_f32

    hip.reciprocal(%ctx) ins(%input : memref<128xf32, 1>)
                         outs(%output : memref<128xf32, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0.000000e+00 : f64) : f64
    // CHECK: %{{.*}} = llvm.mlir.constant(1.000000e+00 : f64) : f64
    // CHECK: %{{.*}} = llvm.mlir.constant(-1.000000e+00 : f64) : f64
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }

  // Test 2: Static 2D f16 - num_elements = 128 * 512, data_type = 1 (f16)
  func.func @reciprocal_static_2d_f16(
      %ctx: !hip.context,
      %input: memref<128x512xf16, 1>,
      %output: memref<128x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @reciprocal_static_2d_f16

    hip.reciprocal(%ctx) ins(%input : memref<128x512xf16, 1>)
                         outs(%output : memref<128x512xf16, 1>)

    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.mlir.constant(512 : i64)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }

  // Test 3: Static 3D bf16 - num_elements = 2 * 3 * 4, data_type = 2 (bf16)
  func.func @reciprocal_static_3d_bf16(
      %ctx: !hip.context,
      %input: memref<2x3x4xbf16, 1>,
      %output: memref<2x3x4xbf16, 1>) {
    // CHECK-LABEL: llvm.func @reciprocal_static_3d_bf16

    hip.reciprocal(%ctx) ins(%input : memref<2x3x4xbf16, 1>)
                         outs(%output : memref<2x3x4xbf16, 1>)

    // CHECK: llvm.mlir.constant(2 : i64)
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }

  // Test 4: Dynamic 2D f32 - num_elements computed at runtime, data_type = 0
  func.func @reciprocal_dynamic_2d_f32(
      %ctx: !hip.context,
      %input: memref<?x?xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @reciprocal_dynamic_2d_f32

    hip.reciprocal(%ctx) ins(%input : memref<?x?xf32, 1>)
                         outs(%output : memref<?x?xf32, 1>)

    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.mul
    // CHECK: llvm.mlir.constant(0 : i64)
    // CHECK: llvm.call @wrap_power({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, f64, f64, f64) -> i32

    return
  }
}
