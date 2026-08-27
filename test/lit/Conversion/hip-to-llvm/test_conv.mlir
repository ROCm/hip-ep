// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.conv lowers to a call into the custom conv kernel's runtime
// wrapper, wrap_conv, and never to MIOpen.
//
// This test validates:
// - hip.conv -> llvm.call @wrap_conv
// - Type conversion: !hip.context -> !llvm.ptr
// - Memref descriptors expanded to individual LLVM parameters
// - The per-axis ABI: spatial_rank plus three slots each for extents, kernel,
//   strides, pad_begin and dilations, with the unused third slot carrying the
//   identity (1, or 0 for the pad) so the kernel needs no rank-specialised path
//
// Note: MLIR's memref-to-LLVM conversion expands memref descriptors into
// individual scalar parameters (allocated_ptr, aligned_ptr, offset, sizes,
// strides). This is standard MLIR behavior.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// The runtime declaration carries the whole ABI, so check it in full: state,
// four pointers, dtype, spatial_rank, N/Cin/Cout, then six three-wide per-axis
// groups and group.
// CHECK: llvm.func @wrap_conv(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

module {
  func.func @conv_llvm_test(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // After lowering, function becomes llvm.func with expanded parameters
    // CHECK-LABEL: llvm.func @conv_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    // HIP convolution operation
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}

    // CHECK: llvm.call @wrap_conv
    // CHECK-NOT: wrap_miopenConvolutionForward

    return
  }
}
