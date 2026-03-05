// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX MaxPoolSingleOut is correctly lowered to hip.maxpool.
//
// Test cases:
// 1. maxpool_basic    — 2x2 pool, stride 2, no padding
// 2. maxpool_padded   — 3x3 pool, stride 1, symmetric padding
// 3. maxpool_default_stride — kernel only, strides default to [1,1]
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - kernel_shape, strides, pads forwarded correctly
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. Basic max pool: 2x2 window, stride 2, no padding
  // --------------------------------------------------------------------------
  func.func @maxpool_basic(%input: tensor<1x64x4x4xf32>) -> tensor<1x64x2x2xf32> {
    %output = "onnx.MaxPoolSingleOut"(%input) {
      kernel_shape = [2, 2],
      strides = [2, 2],
      pads = [0, 0, 0, 0]
    } : (tensor<1x64x4x4xf32>) -> tensor<1x64x2x2xf32>
    return %output : tensor<1x64x2x2xf32>
  }

  // CHECK-LABEL: func.func @maxpool_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x4x4xf32>) -> tensor<1x64x2x2xf32>
  // CHECK: tensor.empty() : tensor<1x64x2x2xf32>
  // CHECK: hip.maxpool(%[[CTX]], %[[IN]], {{.*}}) {kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]}
  // CHECK-SAME: (!hip.context, tensor<1x64x4x4xf32>, tensor<1x64x2x2xf32>) : tensor<1x64x2x2xf32>
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. Max pool with symmetric padding: 3x3 window, stride 1, pad 1
  // --------------------------------------------------------------------------
  func.func @maxpool_padded(%input: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32> {
    %output = "onnx.MaxPoolSingleOut"(%input) {
      kernel_shape = [3, 3],
      strides = [1, 1],
      pads = [1, 1, 1, 1]
    } : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %output : tensor<1x64x8x8xf32>
  }

  // CHECK-LABEL: func.func @maxpool_padded
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
  // CHECK: tensor.empty() : tensor<1x64x8x8xf32>
  // CHECK: hip.maxpool(%[[CTX]], %[[IN]], {{.*}}) {kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. Max pool with default strides (omitted → pattern fills [1,1])
  // --------------------------------------------------------------------------
  func.func @maxpool_default_stride(%input: tensor<1x64x4x4xf32>) -> tensor<1x64x3x3xf32> {
    %output = "onnx.MaxPoolSingleOut"(%input) {
      kernel_shape = [2, 2],
      pads = [0, 0, 0, 0]
    } : (tensor<1x64x4x4xf32>) -> tensor<1x64x3x3xf32>
    return %output : tensor<1x64x3x3xf32>
  }

  // CHECK-LABEL: func.func @maxpool_default_stride
  // CHECK-SAME: !hip.context
  // CHECK: hip.maxpool({{.*}}) {kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]}
  // CHECK-NOT: hip.alloc
}
