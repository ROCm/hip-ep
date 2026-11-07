// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// With --use-dynamic-dispatch, hip.conv lowers to wrap_dd_conv2d (NPU) only
// when DynamicDispatch iconv can run it (include/hip/dynamic_dispatch_support.h):
// f16, N == 1, channel counts that are multiples of 8, no dilation, no
// groups, no bias and no padding. Everything else lowers to the GPU (wrap_conv).
//
// wrap_dd_conv2d signature: state, op_state_slot, input, n, c, h, w, weights,
// k, bias, output, out_h, out_w, kernel_h, kernel_w, stride_h, stride_w,
// pad_top, pad_left, pad_bottom, pad_right, dilation_h, dilation_w, group,
// data_type.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm="use-dynamic-dispatch=true" | FileCheck %s
// With the CoreLib per-op backend also on: it routes other ops, so nothing here changes.
// RUN: hip-mlir-opt %s --convert-hip-to-llvm="use-dynamic-dispatch=true use-ryzenai-corelib=true" | FileCheck %s

module {
  // CHECK-DAG: llvm.func @wrap_dd_conv2d(!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

  // CHECK-LABEL: llvm.func @dd_conv
  // CHECK: llvm.call @wrap_dd_conv2d({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func @dd_conv(%ctx: !hip.context,
                     %input: memref<1x64x56x56xf16, 1>,
                     %weights: memref<64x64x3x3xf16, 1>,
                     %output: memref<1x64x54x54xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x64x56x56xf16, 1>,
                                          memref<64x64x3x3xf16, 1>)
                   outs(%output : memref<1x64x54x54xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // ----- GPU fallbacks -------------------------------------------------------

  // CHECK-LABEL: llvm.func @gpu_conv_bias
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_bias(%ctx: !hip.context,
                           %input: memref<1x64x56x56xf16, 1>,
                           %weights: memref<64x64x3x3xf16, 1>,
                           %bias: memref<64xf16, 1>,
                           %output: memref<1x64x54x54xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights, %bias : memref<1x64x56x56xf16, 1>,
                                                 memref<64x64x3x3xf16, 1>,
                                                 memref<64xf16, 1>)
                   outs(%output : memref<1x64x54x54xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_conv_padding
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_padding(%ctx: !hip.context,
                              %input: memref<1x64x56x56xf16, 1>,
                              %weights: memref<64x64x3x3xf16, 1>,
                              %output: memref<1x64x56x56xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x64x56x56xf16, 1>,
                                          memref<64x64x3x3xf16, 1>)
                   outs(%output : memref<1x64x56x56xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_conv_channels_not_multiple_of_8
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_channels_not_multiple_of_8(
      %ctx: !hip.context,
      %input: memref<1x3x224x224xf16, 1>,
      %weights: memref<64x3x7x7xf16, 1>,
      %output: memref<1x64x109x109xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x3x224x224xf16, 1>,
                                          memref<64x3x7x7xf16, 1>)
                   outs(%output : memref<1x64x109x109xf16, 1>)
                   {kernel_shape = [7, 7], strides = [2, 2],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_conv_depthwise
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_depthwise(%ctx: !hip.context,
                                %input: memref<1x256x28x28xf16, 1>,
                                %weights: memref<256x1x3x3xf16, 1>,
                                %output: memref<1x256x26x26xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x256x28x28xf16, 1>,
                                          memref<256x1x3x3xf16, 1>)
                   outs(%output : memref<1x256x26x26xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 256}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_conv_dilated
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_dilated(%ctx: !hip.context,
                              %input: memref<1x128x32x32xf16, 1>,
                              %weights: memref<128x128x3x3xf16, 1>,
                              %output: memref<1x128x28x28xf16, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x128x32x32xf16, 1>,
                                          memref<128x128x3x3xf16, 1>)
                   outs(%output : memref<1x128x28x28xf16, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [2, 2], group = 1}
    return
  }

  // CHECK-LABEL: llvm.func @gpu_conv_f32
  // CHECK-NOT: wrap_dd_conv2d
  // CHECK: llvm.call @wrap_conv(
  func.func @gpu_conv_f32(%ctx: !hip.context,
                          %input: memref<1x64x56x56xf32, 1>,
                          %weights: memref<64x64x3x3xf32, 1>,
                          %output: memref<1x64x54x54xf32, 1>) {
    hip.conv(%ctx) ins(%input, %weights : memref<1x64x56x56xf32, 1>,
                                          memref<64x64x3x3xf32, 1>)
                   outs(%output : memref<1x64x54x54xf32, 1>)
                   {kernel_shape = [3, 3], strides = [1, 1],
                    pads = [0, 0, 0, 0], dilations = [1, 1], group = 1}
    return
  }
}
