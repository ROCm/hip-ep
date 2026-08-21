// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP convolution operation is correctly lowered to DynamicDispatch
// backend when --use-dynamic-dispatch option is enabled.
//
// This test validates:
// - hip.conv → llvm.call @wrap_dd_conv2d (NPU/IPU backend)
// - Function signature matches DynamicDispatch C ABI
// - 24-param signature: state, op_state_slot, input, n, c, h, w, weights, k,
//                        bias, output, out_h, out_w, kernel_h, kernel_w,
//                        stride_h, stride_w, pad_top, pad_left, pad_bottom,
//                        pad_right, dilation_h, dilation_w, group, data_type
// - Attributes extracted: kernel_shape, strides, pads, dilations, group
// - Input/output dimensions extracted from memref shapes
// - Data type passed as i64 constant
//
// Expected: wrap_dd_conv2d(state, op_state_slot, input_ptr, n, c, h, w,
//             weights_ptr, k, bias_ptr, output_ptr, out_h, out_w, kernel_h,
//             kernel_w, stride_h, stride_w, pad_top, pad_left, pad_bottom,
//             pad_right, dilation_h, dilation_w, group, data_type)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm="use-dynamic-dispatch=true" | FileCheck %s

module {
  // Test 1: Basic Conv2D with standard attributes
  // Input: [1, 3, 224, 224], Weights: [64, 3, 7, 7], Bias: [64]
  // Output: [1, 64, 112, 112]
  // kernel_shape=[7,7], strides=[2,2], pads=[3,3,3,3], dilations=[1,1], group=1
  func.func @dd_conv_basic(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf32, 1>,
      %weights: memref<64x3x7x7xf32, 1>,
      %bias: memref<64xf32, 1>,
      %output: memref<1x64x112x112xf32, 1>) {
    // CHECK-DAG: llvm.func @wrap_dd_conv2d(!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    // CHECK-LABEL: llvm.func @dd_conv_basic
    // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x3x224x224xf32, 1>,
                                                 memref<64x3x7x7xf32, 1>,
                                                 memref<64xf32, 1>)
                   outs(%output : memref<1x64x112x112xf32, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [3, 3, 3, 3], dilations = [1, 1], group = 1}

    return
  }

  // Test 2: Conv2D with no bias (null pointer)
  // Input: [1, 64, 56, 56], Weights: [64, 64, 3, 3]
  // Output: [1, 64, 56, 56] (same padding)
  func.func @dd_conv_no_bias(
      %ctx: !hip.context,
      %input: memref<1x64x56x56xf16, 1>,
      %weights: memref<64x64x3x3xf16, 1>,
      %output: memref<1x64x56x56xf16, 1>) {
    // CHECK-LABEL: llvm.func @dd_conv_no_bias

    hip.conv(%ctx) ins(%input, %weights : memref<1x64x56x56xf16, 1>,
                                          memref<64x64x3x3xf16, 1>)
                   outs(%output : memref<1x64x56x56xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}

    // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 3: Depthwise convolution (group = channels)
  // Input: [1, 256, 28, 28], Weights: [256, 1, 3, 3]
  // Output: [1, 256, 28, 28]
  // group=256 (depthwise)
  func.func @dd_conv_depthwise(
      %ctx: !hip.context,
      %input: memref<1x256x28x28xf32, 1>,
      %weights: memref<256x1x3x3xf32, 1>,
      %output: memref<1x256x28x28xf32, 1>) {
    // CHECK-LABEL: llvm.func @dd_conv_depthwise

    hip.conv(%ctx) ins(%input, %weights : memref<1x256x28x28xf32, 1>,
                                          memref<256x1x3x3xf32, 1>)
                   outs(%output : memref<1x256x28x28xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 256}

    // Verify group=256 passed as i64 constant
    // CHECK: llvm.mlir.constant(256 : i64) : i64
    // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 4: Dilated convolution
  // Input: [1, 128, 32, 32], Weights: [128, 128, 3, 3]
  // Output: [1, 128, 32, 32]
  // dilations=[2, 2]
  func.func @dd_conv_dilated(
      %ctx: !hip.context,
      %input: memref<1x128x32x32xf16, 1>,
      %weights: memref<128x128x3x3xf16, 1>,
      %output: memref<1x128x32x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @dd_conv_dilated

    hip.conv(%ctx) ins(%input, %weights : memref<1x128x32x32xf16, 1>,
                                          memref<128x128x3x3xf16, 1>)
                   outs(%output : memref<1x128x32x32xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [2, 2, 2, 2], dilations = [2, 2], group = 1}

    // Verify dilation_h=2, dilation_w=2 passed as i64 constants
    // CHECK-DAG: llvm.mlir.constant(2 : i64) : i64
    // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 5: Asymmetric padding
  // Input: [1, 64, 100, 100], Weights: [128, 64, 5, 5]
  // Output: [1, 128, 50, 50]
  // pads=[2, 1, 2, 1] (asymmetric)
  func.func @dd_conv_asymmetric_padding(
      %ctx: !hip.context,
      %input: memref<1x64x100x100xf32, 1>,
      %weights: memref<128x64x5x5xf32, 1>,
      %output: memref<1x128x50x50xf32, 1>) {
    // CHECK-LABEL: llvm.func @dd_conv_asymmetric_padding

    hip.conv(%ctx) ins(%input, %weights : memref<1x64x100x100xf32, 1>,
                                          memref<128x64x5x5xf32, 1>)
                   outs(%output : memref<1x128x50x50xf32, 1>)
                   {kernel_shape = [5, 5], strides = [2, 2],
                    pads = [2, 1, 2, 1], dilations = [1, 1], group = 1}

    // Verify asymmetric padding: pad_top=2, pad_left=1, pad_bottom=2, pad_right=1
    // CHECK-DAG: llvm.mlir.constant(2 : i64) : i64
    // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }
}
