// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Conv operation is correctly lowered to hip.conv operation.
//
// This test validates:
// - Basic 2D convolution lowering (onnx.Conv → hip.conv)
// - Attribute preservation (kernel_shape, strides, pads, dilations, group)
// - Type conversion (ONNX tensor ops → HIP memref ops)
// - Proper handling of input, weights, and bias operands
//
// Input: ONNX Conv with standard ResNet-50 first layer configuration
//        (3→64 channels, 7x7 kernel, stride 2, padding 3)
// Expected: hip.conv operation with identical attributes
// ============================================================================

// RUN: hip-opt %s --convert-onnx-to-hip | FileCheck %s

module {
  func.func @conv_test(%input: tensor<1x3x224x224xf32>, %weights: tensor<64x3x7x7xf32>, %bias: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    // After conversion: context added, tensors→memrefs, output arg added, return→i32
    // CHECK-LABEL: func.func @conv_test
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: memref<1x3x224x224xf32, 1>, %[[WEIGHTS:.*]]: memref<64x3x7x7xf32, 1>, %[[BIAS:.*]]: memref<64xf32, 1>, %[[OUTPUT_ARG:.*]]: memref<1x64x112x112xf32, 1>) -> i32

    // ONNX Conv operation
    %output = "onnx.Conv"(%input, %weights, %bias) {
      kernel_shape = [7, 7],
      strides = [2, 2],
      pads = [3, 3, 3, 3],
      dilations = [1, 1],
      group = 1 : si64
    } : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>

    // After conversion: call hip.conv with output arg, return status
    // NOTE: Conv writes directly to output arg (no temp buffer needed)
    // CHECK: hip.conv(%[[CTX]], %[[INPUT]], %[[WEIGHTS]], %[[BIAS]], %[[OUTPUT_ARG]])
    // CHECK-SAME: {dilations = [1, 1], group = 1 : i64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}
    // CHECK-NEXT: %[[STATUS:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: return %[[STATUS]] : i32

    return %output : tensor<1x64x112x112xf32>
  }
}
