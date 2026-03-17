// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP ReLU operation is correctly lowered to LLVM IR with
// MemRef-agnostic runtime wrapper signature.
//
// This test validates:
// - hip.relu lowering to llvm.call @wrap_miopenActivationForward_relu
// - Correct extraction of GPU pointer from MemRef descriptor
// - Correct extraction of tensor dimensions (N, C, H, W)
// - MemRef-agnostic signature: runtime receives GPU pointer + dimensions
//   (NOT MemRef descriptor pointer)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @test_relu_lowering(%ctx: !hip.context,
                                %input: memref<1x64x224x224xf32, 1>,
                                %output: memref<1x64x224x224xf32, 1>) {
    // HIP ReLU operation
    hip.relu(%ctx) ins(%input : memref<1x64x224x224xf32, 1>)
                   outs(%output : memref<1x64x224x224xf32, 1>)

    // After lowering to LLVM: Runtime wrapper should have MemRef-agnostic signature
    // Function declaration appears before function definition in module
    // CHECK: llvm.func @wrap_miopenActivationForward_relu(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> i32

    // Function definition with unpacked MemRef parameters
    // CHECK-LABEL: llvm.func @test_relu_lowering

    // Should cast GPU pointers from address space 1 to address space 0
    // CHECK: llvm.addrspacecast {{.*}} : !llvm.ptr<1> to !llvm.ptr
    // CHECK: llvm.addrspacecast {{.*}} : !llvm.ptr<1> to !llvm.ptr

    // Should create dimension constants (from static shape [1x64x224x224])
    // Input dimensions
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(64 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(224 : i64) : i64
    // Output dimensions (same as input for ReLU)
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(64 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(224 : i64) : i64

    // Should call wrapper with GPU pointers and dimension constants
    // Signature: (state, input_ptr, input_n, input_c, input_h, input_w, output_ptr, output_n, output_c, output_h, output_w)
    // CHECK: llvm.call @wrap_miopenActivationForward_relu({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, i64, i64, i64) -> i32

    return
  }
}
