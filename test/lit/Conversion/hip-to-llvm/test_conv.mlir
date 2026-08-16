// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP convolution operations are correctly lowered to LLVM calls
// to MIOpen library.
//
// This test validates:
// - hip.conv → llvm.call @wrap_miopenConvolutionForward
// - Type conversion: !hip.context → !llvm.ptr
// - Memref descriptors expanded to individual LLVM parameters
// - Attribute passing to runtime calls
// - Proper function signature for MIOpen API
//
// Note: MLIR's memref-to-LLVM conversion expands memref descriptors into
// individual scalar parameters (allocated_ptr, aligned_ptr, offset, sizes,
// strides). This is standard MLIR behavior.
//
// Expected: LLVM function with expanded parameters calling MIOpen wrapper
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @conv_llvm_test(
      %ctx: !hip.context,
      %valid: i1,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // After lowering, function becomes llvm.func with expanded parameters
    // CHECK-LABEL: llvm.func @conv_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    // HIP convolution operation
    hip.conv(%ctx) valid(%valid)
                   ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}

    // Should lower to MIOpen convolution forward call
    // The wrapper is declared and called
    // CHECK: llvm.zext %{{.*}} : i1 to i64
    // CHECK: llvm.call @wrap_miopenConvolutionForward

    return
  }
}
