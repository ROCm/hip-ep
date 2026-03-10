// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP cast operation is correctly lowered to LLVM call
// to wrap_miopenCast runtime function with both static and dynamic shapes.
//
// This test validates:
// - hip.cast → llvm.call @wrap_miopenCast
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: num_elements computed at compile time
// - Dynamic shapes: num_elements computed at runtime via extractvalue
// - Proper src_data_type and dst_data_type parameters
//
// Expected: wrap_miopenCast(state, input_ptr, output_ptr,
//                           num_elements, src_data_type, dst_data_type)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 2D tensor (f32 -> f16)
  func.func @cast_static_2d_f32_to_f16(
      %ctx: !hip.context,
      %input: memref<256x512xf32, 1>,
      %output: memref<256x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @cast_static_2d_f32_to_f16
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.cast(%ctx) ins(%input : memref<256x512xf32, 1>)
                   outs(%output : memref<256x512xf16, 1>)

    // CHECK: llvm.call @wrap_miopenCast(%[[CTX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 2: Static 3D tensor (f16 -> f32)
  func.func @cast_static_3d_f16_to_f32(
      %ctx: !hip.context,
      %input: memref<1x128x512xf16, 1>,
      %output: memref<1x128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @cast_static_3d_f16_to_f32
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.cast(%ctx) ins(%input : memref<1x128x512xf16, 1>)
                   outs(%output : memref<1x128x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenCast(%[[CTX]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic shapes (f32 -> f16)
  func.func @cast_dynamic_f32_to_f16(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %output: memref<?x?x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @cast_dynamic_f32_to_f16
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.cast(%ctx) ins(%input : memref<?x?x512xf32, 1>)
                   outs(%output : memref<?x?x512xf16, 1>)

    // Verify dynamic shape computation
    // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[DIM0:.*]] = llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: %[[PROD1:.*]] = llvm.mul %[[ONE]], %[[DIM0]] : i64
    // CHECK: %[[DIM1:.*]] = llvm.extractvalue %{{.*}}[3, 1]
    // CHECK: %[[PROD2:.*]] = llvm.mul %[[PROD1]], %[[DIM1]] : i64
    // CHECK: %[[DIM2:.*]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[NUM_ELEMENTS:.*]] = llvm.mul %[[PROD2]], %[[DIM2]] : i64

    // Verify data type constants (f32=0, f16=1)
    // CHECK: %[[SRC_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[DST_TYPE:.*]] = llvm.mlir.constant(1 : i64) : i64

    // CHECK: llvm.call @wrap_miopenCast(%[[CTX]], %{{.*}}, %{{.*}}, %[[NUM_ELEMENTS]], %[[SRC_TYPE]], %[[DST_TYPE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }
}
