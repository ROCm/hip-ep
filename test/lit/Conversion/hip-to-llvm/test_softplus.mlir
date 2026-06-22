// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP softplus operation is correctly lowered to LLVM call
// to wrap_miopenActivationForward runtime function with activation_mode=3.
//
// This test validates:
// - hip.softplus → llvm.call @wrap_miopenActivationForward
// - Type conversion: !hip.context → !llvm.ptr
// - Static shapes: num_elements computed with mul operations
// - Dynamic shapes: num_elements computed at runtime via extractvalue
// - Proper function signature for runtime API
//
// Expected: wrap_miopenActivationForward(state, input_ptr, output_ptr,
//                                         num_elements, data_type, activation_mode)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Static 1D tensor (f32)
  func.func @softplus_static_1d_f32_test(
      %ctx: !hip.context,
      %input: memref<128xf32, 1>,
      %output: memref<128xf32, 1>) {
    // CHECK-LABEL: llvm.func @softplus_static_1d_f32_test

    hip.softplus(%ctx) ins(%input : memref<128xf32, 1>)
                       outs(%output : memref<128xf32, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 2: Static 3D tensor (f32) - data_type 0 = f32
  func.func @softplus_static_3d_f32_test(
      %ctx: !hip.context,
      %input: memref<2x3x4xf32, 1>,
      %output: memref<2x3x4xf32, 1>) {
    // CHECK-LABEL: llvm.func @softplus_static_3d_f32_test

    hip.softplus(%ctx) ins(%input : memref<2x3x4xf32, 1>)
                       outs(%output : memref<2x3x4xf32, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(4 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 3: Static 2D tensor (f16) - data_type 1 = f16
  func.func @softplus_static_2d_f16_test(
      %ctx: !hip.context,
      %input: memref<256x512xf16, 1>,
      %output: memref<256x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @softplus_static_2d_f16_test

    hip.softplus(%ctx) ins(%input : memref<256x512xf16, 1>)
                       outs(%output : memref<256x512xf16, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(256 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 4: Static 2D tensor (bf16) - data_type 2 = bf16
  func.func @softplus_static_2d_bf16_test(
      %ctx: !hip.context,
      %input: memref<128x256xbf16, 1>,
      %output: memref<128x256xbf16, 1>) {
    // CHECK-LABEL: llvm.func @softplus_static_2d_bf16_test

    hip.softplus(%ctx) ins(%input : memref<128x256xbf16, 1>)
                       outs(%output : memref<128x256xbf16, 1>)

    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(128 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(256 : i64) : i64
    // CHECK: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 5: Dynamic 2D tensor
  func.func @softplus_dynamic_2d_test(
      %ctx: !hip.context,
      %input: memref<?x?xf32, 1>,
      %output: memref<?x?xf32, 1>) {
    // CHECK-LABEL: llvm.func @softplus_dynamic_2d_test

    hip.softplus(%ctx) ins(%input : memref<?x?xf32, 1>)
                       outs(%output : memref<?x?xf32, 1>)

    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: %{{.*}} = llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: %{{.*}} = llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(0 : i64) : i64
    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }

  // Test 6: Partially dynamic 3D tensor
  func.func @softplus_partial_dynamic_test(
      %ctx: !hip.context,
      %input: memref<?x?x512xf16, 1>,
      %output: memref<?x?x512xf16, 1>) {
    // CHECK-LABEL: llvm.func @softplus_partial_dynamic_test

    hip.softplus(%ctx) ins(%input : memref<?x?x512xf16, 1>)
                       outs(%output : memref<?x?x512xf16, 1>)

    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: %{{.*}} = llvm.extractvalue %{{.*}}[3, 0]
    // CHECK-DAG: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: %{{.*}} = llvm.extractvalue %{{.*}}[3, 1]
    // CHECK-DAG: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(512 : i64) : i64
    // CHECK-DAG: %{{.*}} = llvm.mul %{{.*}}, %{{.*}} : i64
    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: %{{.*}} = llvm.mlir.constant(3 : i64) : i64
    // CHECK: %{{.*}} = llvm.call @wrap_miopenActivationForward(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32

    return
  }
}
