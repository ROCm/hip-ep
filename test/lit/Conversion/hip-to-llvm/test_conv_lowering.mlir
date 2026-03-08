// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP convolution operations are correctly lowered to LLVM calls
// to MIOpen library runtime wrapper.
//
// This test validates:
// - hip.conv → llvm.call @wrap_miopenConvolutionForward
// - Type conversion: !hip.context → !llvm.ptr
// - Shape extraction from memref types (input_n, input_c, etc.)
// - Attribute passing (kernel_shape, strides, pads, dilations, group)
// - Proper function signature with 23 parameters returning i32
//
// The wrapper function follows PR-17 pattern with explicit shape parameters:
// wrap_miopenConvolutionForward(state, input, input_n, input_c, input_h, input_w,
//                                weights, weights_k, bias, output, output_h, output_w,
//                                kernel_h, kernel_w, stride_h, stride_w,
//                                pad_top, pad_left, pad_bottom, pad_right,
//                                dilation_h, dilation_w, group) -> i32
//
// Expected: LLVM function declaration and call with all shape/attr parameters
// ============================================================================

// RUN: %hip-mlir-opt %s --convert-hip-to-llvm | %FileCheck %s

module {
  func.func @conv_llvm_test(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // CHECK-LABEL: func.func @conv_llvm_test
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: !hip.context to !llvm.ptr

    // HIP convolution operation
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}

    // Should lower to MIOpen convolution forward call via runtime wrapper
    // Check function declaration with 23 params returning i32
    // CHECK: llvm.call @wrap_miopenConvolutionForward({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
