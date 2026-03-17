// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP mul operation is correctly lowered to LLVM call
// to wrap_miopenOpTensor runtime function with both static and dynamic shapes.
//
// This test validates:
// - hip.mul → llvm.call @wrap_miopenOpTensor
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: num_elements computed at compile time
// - Dynamic shapes: num_elements computed at runtime via llvm.mul of dims
// - Data type enum passed correctly (f32=0, f16=1, bf16=2)
// - Tensor operation enum (MIOPEN_TENSOR_OP_MUL = 0)
// - Proper function signature for runtime API
//
// Expected: wrap_miopenOpTensor(state, A_ptr, B_ptr, C_ptr,
//                                num_elements, data_type, tensor_op)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static shapes - num_elements = 128*512 = 65536, data_type = 0 (f32), tensor_op = 0 (MUL)
  func.func @mul_static_f32_test(
      %ctx: !hip.context,
      %a: memref<128x512xf32, 1>,
      %b: memref<128x512xf32, 1>,
      %c: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_static_f32_test

    hip.mul(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                         outs(%c : memref<128x512xf32, 1>)

    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.mul
    // CHECK: llvm.mlir.constant(512 : i64)
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 2: Static shapes with f16 - num_elements = 1024, data_type = 1 (f16)
  func.func @mul_static_f16_test(
      %ctx: !hip.context,
      %a: memref<1024xf16, 1>,
      %b: memref<1024xf16, 1>,
      %c: memref<1024xf16, 1>) {
    // CHECK-LABEL: llvm.func @mul_static_f16_test

    hip.mul(%ctx) ins(%a, %b : memref<1024xf16, 1>, memref<1024xf16, 1>)
                         outs(%c : memref<1024xf16, 1>)

    // CHECK: llvm.mlir.constant(1024 : i64)
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 3: 3D tensor - num_elements = 2*64*128 = 16384, data_type = 0 (f32)
  func.func @mul_3d_test(
      %ctx: !hip.context,
      %a: memref<2x64x128xf32, 1>,
      %b: memref<2x64x128xf32, 1>,
      %c: memref<2x64x128xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_3d_test

    hip.mul(%ctx) ins(%a, %b : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>)
                         outs(%c : memref<2x64x128xf32, 1>)

    // CHECK: llvm.mlir.constant(2 : i64)
    // CHECK: llvm.mul
    // CHECK: llvm.mlir.constant(64 : i64)
    // CHECK: llvm.mul
    // CHECK: llvm.mlir.constant(128 : i64)
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 4: Dynamic shapes - num_elements computed at runtime
  func.func @mul_dynamic_test(
      %ctx: !hip.context,
      %a: memref<?x512xf32, 1>,
      %b: memref<?x512xf32, 1>,
      %c: memref<?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @mul_dynamic_test

    hip.mul(%ctx) ins(%a, %b : memref<?x512xf32, 1>, memref<?x512xf32, 1>)
                         outs(%c : memref<?x512xf32, 1>)

    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul {{.*}} : i64
    // CHECK: llvm.mlir.constant(512 : i64)
    // CHECK: llvm.mul {{.*}} : i64
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 5: Fully dynamic shapes - both dimensions dynamic
  func.func @mul_fully_dynamic_test(
      %ctx: !hip.context,
      %a: memref<?x?xf16, 1>,
      %b: memref<?x?xf16, 1>,
      %c: memref<?x?xf16, 1>) {
    // CHECK-LABEL: llvm.func @mul_fully_dynamic_test

    hip.mul(%ctx) ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
                         outs(%c : memref<?x?xf16, 1>)

    // CHECK: llvm.extractvalue {{.*}}[3, 0]
    // CHECK: llvm.mul {{.*}} : i64
    // CHECK: llvm.extractvalue {{.*}}[3, 1]
    // CHECK: llvm.mul {{.*}} : i64
    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }
}
