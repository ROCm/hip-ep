// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sigmoid is correctly lowered to hip.sigmoid in tensor-first mode.
//
// Test cases:
// 1. sigmoid_2d          — 2D tensor sigmoid activation
// 2. sigmoid_3d          — 3D tensor sigmoid activation
// 3. sigmoid_4d          — 4D tensor sigmoid activation (typical CNN layer)
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - proper shape preservation
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. 2D tensor sigmoid
  // --------------------------------------------------------------------------
  func.func @sigmoid_2d(%input: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %output = "onnx.Sigmoid"(%input) : (tensor<128x256xf32>) -> tensor<128x256xf32>
    return %output : tensor<128x256xf32>
  }

  // CHECK-LABEL: func.func @sigmoid_2d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<128x256xf32>) -> tensor<128x256xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128x256xf32>
  // CHECK: %[[OUT:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[IN]] : tensor<128x256xf32>) outs(%[[INIT]] : tensor<128x256xf32>) -> tensor<128x256xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. 3D tensor sigmoid (batch, sequence, features)
  // --------------------------------------------------------------------------
  func.func @sigmoid_3d(%input: tensor<8x128x512xf32>) -> tensor<8x128x512xf32> {
    %output = "onnx.Sigmoid"(%input) : (tensor<8x128x512xf32>) -> tensor<8x128x512xf32>
    return %output : tensor<8x128x512xf32>
  }

  // CHECK-LABEL: func.func @sigmoid_3d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<8x128x512xf32>) -> tensor<8x128x512xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8x128x512xf32>
  // CHECK: %[[OUT:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[IN]] : tensor<8x128x512xf32>) outs(%[[INIT]] : tensor<8x128x512xf32>) -> tensor<8x128x512xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. 4D tensor sigmoid (typical CNN activation)
  // --------------------------------------------------------------------------
  func.func @sigmoid_4d(%input: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> {
    %output = "onnx.Sigmoid"(%input) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %output : tensor<1x64x56x56xf32>
  }

  // CHECK-LABEL: func.func @sigmoid_4d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x64x56x56xf32>
  // CHECK: %[[OUT:.*]] = hip.sigmoid(%[[CTX]]) ins(%[[IN]] : tensor<1x64x56x56xf32>) outs(%[[INIT]] : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc
}
