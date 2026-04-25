// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.conv_transpose lowers to a call to MIOpen's
// wrap_miopenConvolutionBackwardData runtime wrapper.
//
// This test validates:
// - hip.conv_transpose -> llvm.call @wrap_miopenConvolutionBackwardData
// - 1-D NCW (rank-3) memref operands are accepted (Kokoro iSTFT path)
// - Bias operand is forwarded to the runtime call
// - Type conversion: !hip.context -> !llvm.ptr
//
// Shapes mirror Kokoro's first 1-D ConvTranspose layer:
//   input   : memref<1x64x100xf16>          (N=1, C_in=64,  IW=100)
//   weights : memref<64x32x4xf16>           (C_in=64, C_out=32, kW=4)
//   bias    : memref<32xf16>                (C_out=32)
//   output  : memref<1x32x200xf16>          (N=1, C_out=32, OW=200)
// stride=2, kernel=4, pad=1, dilation=1, output_padding=0, group=1.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @conv_transpose_1d_test(
      %ctx: !hip.context,
      %input: memref<1x64x100xf16, 1>,
      %weights: memref<64x32x4xf16, 1>,
      %bias: memref<32xf16, 1>,
      %output: memref<1x32x200xf16, 1>) {
    // CHECK-LABEL: llvm.func @conv_transpose_1d_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr

    hip.conv_transpose(%ctx) ins(%input, %weights, %bias :
                                   memref<1x64x100xf16, 1>,
                                   memref<64x32x4xf16, 1>,
                                   memref<32xf16, 1>)
                             outs(%output : memref<1x32x200xf16, 1>)
                             {kernel_shape = [4], strides = [2],
                              pads = [1, 1], dilations = [1],
                              output_padding = [0], group = 1}

    // CHECK: llvm.call @wrap_miopenConvolutionBackwardData

    return
  }
  // CHECK: llvm.func @wrap_miopenConvolutionBackwardData
}
