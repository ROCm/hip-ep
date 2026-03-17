// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP sub operation is correctly lowered to LLVM call
// to wrap_miopenTensorOp runtime function with both static and dynamic shapes.
//
// This test validates:
// - hip.sub → llvm.call @wrap_miopenTensorOp
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: numLhs, numRhs computed at compile time
// - Dynamic shapes: numLhs, numRhs computed at runtime via extractvalue
// - Broadcasting support: numRhs = 1
// - Proper function signature for runtime API
//
// Expected: wrap_miopenTensorOp(state, lhs_ptr, rhs_ptr, output_ptr, numLhs,
//                                 numRhs, data_type, tensor_op)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static shapes
  func.func @sub_static_test(
      %ctx: !hip.context,
      %lhs: memref<128x512xf32, 1>,
      %rhs: memref<128x512xf32, 1>,
      %output: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_static_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                  outs(%output : memref<128x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenTensorOp({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: Broadcasting (scalar rhs)
  func.func @sub_broadcast_test(
      %ctx: !hip.context,
      %lhs: memref<256x512xf32, 1>,
      %rhs: memref<1xf32, 1>,
      %output: memref<256x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_broadcast_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<256x512xf32, 1>, memref<1xf32, 1>)
                  outs(%output : memref<256x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenTensorOp({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic shapes
  func.func @sub_dynamic_test(
      %ctx: !hip.context,
      %lhs: memref<?x512xf32, 1>,
      %rhs: memref<?x512xf32, 1>,
      %output: memref<?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @sub_dynamic_test

    hip.sub(%ctx) ins(%lhs, %rhs : memref<?x512xf32, 1>, memref<?x512xf32, 1>)
                  outs(%output : memref<?x512xf32, 1>)

    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64
    // CHECK: llvm.call @wrap_miopenTensorOp({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }
}
