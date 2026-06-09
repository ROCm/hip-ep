// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP sub operation is correctly lowered to LLVM call
// to wrap_elementwise_sub runtime function with both static and dynamic shapes.
//
// This test validates:
// - hip.sub → llvm.call @wrap_elementwise_sub
// - Type conversion: !hip.context → !llvm.ptr
// - 4D broadcast descriptor: lhs/rhs/out shapes passed as 12 i64 dims
// - Proper function signature for runtime API
//
// Expected: wrap_elementwise_sub(state, lhs_ptr, rhs_ptr, output_ptr,
//                                 lhs_n, lhs_c, lhs_h, lhs_w,
//                                 rhs_n, rhs_c, rhs_h, rhs_w,
//                                 out_n, out_c, out_h, out_w, data_type)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static shapes - num_elements = 128*512 = 65536, elem_size = 4 (f32)
  func.func @sub_static_test(
      %ctx: !hip.context,
      %lhs: memref<128x512xf32, 1>,
      %rhs: memref<128x512xf32, 1>,
      %output: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_static_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%output : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: 1D tensor - num_elements = 1024, elem_size = 2 (f16)
  func.func @sub_1d_test(
      %ctx: !hip.context,
      %lhs: memref<1024xf16, 1>,
      %rhs: memref<1024xf16, 1>,
      %output: memref<1024xf16, 1>) {
    // CHECK-LABEL: llvm.func @sub_1d_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<1024xf16, 1>, memref<1024xf16, 1>)
                  outs(%output : memref<1024xf16, 1>)

    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic shapes - num_elements computed at runtime
  func.func @sub_dynamic_test(
      %ctx: !hip.context,
      %lhs: memref<?x512xf32, 1>,
      %rhs: memref<?x512xf32, 1>,
      %output: memref<?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_dynamic_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<?x512xf32, 1>, memref<?x512xf32, 1>)
                  outs(%output : memref<?x512xf32, 1>)

    // CHECK: llvm.call @wrap_elementwise_sub({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
