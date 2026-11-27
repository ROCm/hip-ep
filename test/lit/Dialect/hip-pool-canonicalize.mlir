// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize %s | FileCheck %s

// What this file tests
// --------------------
// `PoolOp::getCanonicalizationPatterns`: an average hip.pool whose one window
// covers the whole input becomes hip.reduce_mean over the spatial dims. See
// lib/Dialect/IR/HipPoolCanonicalize.cpp.
//
// A global average pool (squeeze-and-excitation, a classifier head) is often
// spelled as a pool with kernel = input extent, or with a larger kernel and
// ceil_mode. Either way the result is the mean of the H * W inputs, which the
// reduction computes in one pass. The negative cases are windows that do not
// cover the input, padding (it changes the count_include_pad divisor), other
// pool modes, and inputs the rewrite cannot prove or reduce_mean cannot take.

module {
  // --------------------------------------------------------------------------
  // Kernel = input extent, dynamic batch, f16: the init is reused.
  // --------------------------------------------------------------------------
  func.func @whole_input(%ctx: !hip.context, %x: tensor<?x96x56x56xf16>,
      %init: tensor<?x96x1x1xf16>) -> tensor<?x96x1x1xf16> {
    %y = hip.pool(%ctx) ins(%x : tensor<?x96x56x56xf16>)
        outs(%init : tensor<?x96x1x1xf16>)
        {pool_mode = 0 : i64, kernel_shape = [56, 56], strides = [56, 56],
         pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64, count_include_pad = 1 : i64}
        : tensor<?x96x1x1xf16>
    return %y : tensor<?x96x1x1xf16>
  }

  // CHECK-LABEL: func.func @whole_input
  // CHECK-SAME: %[[X:[^:]*]]: tensor<?x96x56x56xf16>, %[[I:[^:]*]]: tensor<?x96x1x1xf16>
  // CHECK: %[[AXES:.*]] = arith.constant dense<[2, 3]> : tensor<2xi64>
  // CHECK: %[[Y:.*]] = hip.reduce_mean(%{{.*}}) ins(%[[X]], %[[AXES]] : tensor<?x96x56x56xf16>, tensor<2xi64>) outs(%[[I]] : tensor<?x96x1x1xf16>)
  // keepdims = 1 and noop_with_empty_axes = 0 are the defaults, not printed.
  // CHECK-SAME: {reduced_axes = array<i64: 2, 3>}
  // CHECK-NOT: hip.pool
  // CHECK: return %[[Y]]

  // --------------------------------------------------------------------------
  // Kernel larger than the input (ceil_mode, count_include_pad = 1): still
  // the mean of the 7 x 7 inputs, the divisor ONNX Runtime and PyTorch use.
  // --------------------------------------------------------------------------
  func.func @kernel_larger_than_input(%ctx: !hip.context,
      %x: tensor<1x1280x7x7xf16>) -> tensor<1x1280x1x1xf16> {
    %init = tensor.empty() : tensor<1x1280x1x1xf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x1280x7x7xf16>)
        outs(%init : tensor<1x1280x1x1xf16>)
        {pool_mode = 0 : i64, kernel_shape = [1280, 1280],
         strides = [1280, 1280], pads = [0, 0, 0, 0], dilations = [1, 1],
         ceil_mode = 1 : i64, storage_order = 0 : i64,
         count_include_pad = 1 : i64} : tensor<1x1280x1x1xf16>
    return %y : tensor<1x1280x1x1xf16>
  }

  // CHECK-LABEL: func.func @kernel_larger_than_input
  // CHECK: hip.reduce_mean
  // CHECK-SAME: reduced_axes = array<i64: 2, 3>
  // CHECK-NOT: hip.pool

  // --------------------------------------------------------------------------
  // One spatial axis, f32, count_include_pad = 0.
  // --------------------------------------------------------------------------
  func.func @whole_input_1d(%ctx: !hip.context, %x: tensor<2x8x16xf32>)
      -> tensor<2x8x1xf32> {
    %init = tensor.empty() : tensor<2x8x1xf32>
    %y = hip.pool(%ctx) ins(%x : tensor<2x8x16xf32>)
        outs(%init : tensor<2x8x1xf32>)
        {pool_mode = 0 : i64, kernel_shape = [16], strides = [1],
         pads = [0, 0], dilations = [1], ceil_mode = 0 : i64,
         storage_order = 0 : i64} : tensor<2x8x1xf32>
    return %y : tensor<2x8x1xf32>
  }

  // CHECK-LABEL: func.func @whole_input_1d
  // CHECK: %[[AXES:.*]] = arith.constant dense<2> : tensor<1xi64>
  // CHECK: hip.reduce_mean(%{{.*}}) ins(%{{.*}}, %[[AXES]] : tensor<2x8x16xf32>, tensor<1xi64>) outs(%{{.*}} : tensor<2x8x1xf32>)
  // CHECK-SAME: reduced_axes = array<i64: 2>
  // CHECK-NOT: hip.pool

  // --------------------------------------------------------------------------
  // Must stay a pool: a window smaller than the input, a padded whole-input
  // window, a max pool, a dynamic spatial extent, bf16, and a dilated window.
  // --------------------------------------------------------------------------
  func.func @partial_window(%ctx: !hip.context, %x: tensor<1x4x8x8xf16>)
      -> tensor<1x4x2x2xf16> {
    %init = tensor.empty() : tensor<1x4x2x2xf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x8x8xf16>)
        outs(%init : tensor<1x4x2x2xf16>)
        {pool_mode = 0 : i64, kernel_shape = [4, 4], strides = [4, 4],
         pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64} : tensor<1x4x2x2xf16>
    return %y : tensor<1x4x2x2xf16>
  }

  // CHECK-LABEL: func.func @partial_window
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean

  func.func @padded(%ctx: !hip.context, %x: tensor<1x4x6x6xf16>)
      -> tensor<1x4x1x1xf16> {
    %init = tensor.empty() : tensor<1x4x1x1xf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x6x6xf16>)
        outs(%init : tensor<1x4x1x1xf16>)
        {pool_mode = 0 : i64, kernel_shape = [8, 8], strides = [8, 8],
         pads = [1, 1, 1, 1], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64, count_include_pad = 1 : i64}
        : tensor<1x4x1x1xf16>
    return %y : tensor<1x4x1x1xf16>
  }

  // CHECK-LABEL: func.func @padded
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean

  func.func @max_pool(%ctx: !hip.context, %x: tensor<1x4x8x8xf16>)
      -> tensor<1x4x1x1xf16> {
    %init = tensor.empty() : tensor<1x4x1x1xf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x8x8xf16>)
        outs(%init : tensor<1x4x1x1xf16>)
        {pool_mode = 1 : i64, kernel_shape = [8, 8], strides = [8, 8],
         pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64} : tensor<1x4x1x1xf16>
    return %y : tensor<1x4x1x1xf16>
  }

  // CHECK-LABEL: func.func @max_pool
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean

  func.func @dynamic_spatial(%ctx: !hip.context, %x: tensor<1x4x?x8xf16>,
      %init: tensor<1x4x1x1xf16>) -> tensor<1x4x1x1xf16> {
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x?x8xf16>)
        outs(%init : tensor<1x4x1x1xf16>)
        {pool_mode = 0 : i64, kernel_shape = [8, 8], strides = [8, 8],
         pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64} : tensor<1x4x1x1xf16>
    return %y : tensor<1x4x1x1xf16>
  }

  // CHECK-LABEL: func.func @dynamic_spatial
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean

  func.func @bf16(%ctx: !hip.context, %x: tensor<1x4x8x8xbf16>)
      -> tensor<1x4x1x1xbf16> {
    %init = tensor.empty() : tensor<1x4x1x1xbf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x8x8xbf16>)
        outs(%init : tensor<1x4x1x1xbf16>)
        {pool_mode = 0 : i64, kernel_shape = [8, 8], strides = [8, 8],
         pads = [0, 0, 0, 0], dilations = [1, 1], ceil_mode = 0 : i64,
         storage_order = 0 : i64} : tensor<1x4x1x1xbf16>
    return %y : tensor<1x4x1x1xbf16>
  }

  // CHECK-LABEL: func.func @bf16
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean

  func.func @dilated(%ctx: !hip.context, %x: tensor<1x4x8x8xf16>)
      -> tensor<1x4x1x1xf16> {
    %init = tensor.empty() : tensor<1x4x1x1xf16>
    %y = hip.pool(%ctx) ins(%x : tensor<1x4x8x8xf16>)
        outs(%init : tensor<1x4x1x1xf16>)
        {pool_mode = 0 : i64, kernel_shape = [8, 8], strides = [8, 8],
         pads = [0, 0, 0, 0], dilations = [2, 2], ceil_mode = 1 : i64,
         storage_order = 0 : i64} : tensor<1x4x1x1xf16>
    return %y : tensor<1x4x1x1xf16>
  }

  // CHECK-LABEL: func.func @dilated
  // CHECK: hip.pool
  // CHECK-NOT: hip.reduce_mean
}
