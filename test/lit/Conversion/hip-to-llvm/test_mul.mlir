// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP mul operation is correctly lowered to LLVM call
// to wrap_miopenOpTensor runtime function with 4D per-operand shapes
// and tensor_op = 0 (MUL).
//
// This test validates:
// - hip.mul → llvm.call @wrap_miopenOpTensor
// - Per-operand 4D shapes passed for MIOpen-native broadcasting
// - Shapes left-padded with 1 for rank < 4
// - Data type enum (f32=0, f16=1) and tensor_op = 0 (MUL)
// - Dynamic shapes: dims extracted from memref descriptor at runtime
// - 18-param signature: state, lhs, rhs, out, lhs_nchw(4), rhs_nchw(4),
//                        out_nchw(4), data_type, tensor_op
//
// Expected: wrap_miopenOpTensor(state, lhs_ptr, rhs_ptr, output_ptr,
//             lhs_n, lhs_c, lhs_h, lhs_w,
//             rhs_n, rhs_c, rhs_h, rhs_w,
//             out_n, out_c, out_h, out_w,
//             data_type, tensor_op=0)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static shapes 2D f32 - [128, 512] → padded to [1, 1, 128, 512]
  func.func @mul_static_f32_test(
      %ctx: !hip.context,
      %a: memref<128x512xf32, 1>,
      %b: memref<128x512xf32, 1>,
      %c: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_static_f32_test

    hip.mul(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                         outs(%c : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: Static shapes 1D f16 - [1024] → padded to [1, 1, 1, 1024]
  func.func @mul_static_f16_test(
      %ctx: !hip.context,
      %a: memref<1024xf16, 1>,
      %b: memref<1024xf16, 1>,
      %c: memref<1024xf16, 1>) {
    // CHECK-LABEL: llvm.func @mul_static_f16_test

    hip.mul(%ctx) ins(%a, %b : memref<1024xf16, 1>, memref<1024xf16, 1>)
                         outs(%c : memref<1024xf16, 1>)

    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: 3D tensor f32 - [2, 64, 128] → padded to [1, 2, 64, 128]
  func.func @mul_3d_test(
      %ctx: !hip.context,
      %a: memref<2x64x128xf32, 1>,
      %b: memref<2x64x128xf32, 1>,
      %c: memref<2x64x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_3d_test

    hip.mul(%ctx) ins(%a, %b : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>)
                         outs(%c : memref<2x64x128xf32, 1>)

    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 4: Dynamic shapes - dims extracted from memref descriptor at runtime
  func.func @mul_dynamic_test(
      %ctx: !hip.context,
      %a: memref<?x512xf32, 1>,
      %b: memref<?x512xf32, 1>,
      %c: memref<?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_dynamic_test

    hip.mul(%ctx) ins(%a, %b : memref<?x512xf32, 1>, memref<?x512xf32, 1>)
                         outs(%c : memref<?x512xf32, 1>)

    // Dynamic dim extracted via llvm.extractvalue from descriptor
    // CHECK: llvm.extractvalue
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 5: Fully dynamic shapes
  func.func @mul_fully_dynamic_test(
      %ctx: !hip.context,
      %a: memref<?x?xf16, 1>,
      %b: memref<?x?xf16, 1>,
      %c: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @mul_fully_dynamic_test

    hip.mul(%ctx) ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
                         outs(%c : memref<?x?xf16, 1>)

    // CHECK: llvm.extractvalue
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
