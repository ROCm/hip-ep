// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP tanh operation is correctly lowered to LLVM call
// to wrap_miopenActivationForward runtime function with both static and
// dynamic shapes.
//
// This test validates:
// - hip.tanh → llvm.call @wrap_miopenActivationForward
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: num_elements computed at compile time
// - Dynamic shapes: num_elements computed at runtime via extractvalue
// - Proper function signature for runtime API
//
// Expected: wrap_miopenActivationForward(state, input_ptr, output_ptr,
//                                         num_elements, data_type, activation_mode)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 3D tensor
  func.func @tanh_static_3d_test(
      %ctx: !hip.context,
      %input: memref<1x128x512xf32, 1>,
      %output: memref<1x128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @tanh_static_3d_test

    hip.tanh(%ctx) ins(%input : memref<1x128x512xf32, 1>)
                   outs(%output : memref<1x128x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenActivationForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 2: Static 2D tensor
  func.func @tanh_static_2d_test(
      %ctx: !hip.context,
      %input: memref<256x512xf32, 1>,
      %output: memref<256x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @tanh_static_2d_test

    hip.tanh(%ctx) ins(%input : memref<256x512xf32, 1>)
                   outs(%output : memref<256x512xf32, 1>)

    // CHECK: llvm.call @wrap_miopenActivationForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 3: Dynamic shapes
  func.func @tanh_dynamic_test(
      %ctx: !hip.context,
      %input: memref<?x?x512xf32, 1>,
      %output: memref<?x?x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @tanh_dynamic_test

    hip.tanh(%ctx) ins(%input : memref<?x?x512xf32, 1>)
                   outs(%output : memref<?x?x512xf32, 1>)

    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(0 : i64) : i64
    // CHECK: llvm.call @wrap_miopenActivationForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }
}
