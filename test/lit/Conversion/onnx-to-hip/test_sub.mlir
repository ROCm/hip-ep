// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sub is correctly lowered to hip.sub in tensor-first mode.
//
// Test cases:
// 1. sub_2d          — 2D tensor element-wise subtraction
// 2. sub_3d          — 3D tensor subtraction
// 3. sub_broadcast   — Subtraction with broadcasting
//
// All cases assert:
// - context argument prepended
// - tensor.empty() for output init (no hip.alloc)
// - proper shape preservation
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // 1. 2D tensor subtraction
  // --------------------------------------------------------------------------
  func.func @sub_2d(%a: tensor<128x256xf32>, %b: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %output = "onnx.Sub"(%a, %b) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
    return %output : tensor<128x256xf32>
  }

  // CHECK-LABEL: func.func @sub_2d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<128x256xf32>, %[[B:.*]]: tensor<128x256xf32>) -> tensor<128x256xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128x256xf32>
  // CHECK: %[[OUT:.*]] = hip.sub(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<128x256xf32>, tensor<128x256xf32>) outs(%[[INIT]] : tensor<128x256xf32>) : tensor<128x256xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 2. 3D tensor subtraction
  // --------------------------------------------------------------------------
  func.func @sub_3d(%a: tensor<8x128x512xf32>, %b: tensor<8x128x512xf32>) -> tensor<8x128x512xf32> {
    %output = "onnx.Sub"(%a, %b) : (tensor<8x128x512xf32>, tensor<8x128x512xf32>) -> tensor<8x128x512xf32>
    return %output : tensor<8x128x512xf32>
  }

  // CHECK-LABEL: func.func @sub_3d
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<8x128x512xf32>, %[[B:.*]]: tensor<8x128x512xf32>) -> tensor<8x128x512xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<8x128x512xf32>
  // CHECK: %[[OUT:.*]] = hip.sub(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<8x128x512xf32>, tensor<8x128x512xf32>) outs(%[[INIT]] : tensor<8x128x512xf32>) : tensor<8x128x512xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // 3. Subtraction with broadcasting (scalar)
  // --------------------------------------------------------------------------
  func.func @sub_broadcast(%a: tensor<128x256xf32>, %b: tensor<1xf32>) -> tensor<128x256xf32> {
    %output = "onnx.Sub"(%a, %b) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
    return %output : tensor<128x256xf32>
  }

  // CHECK-LABEL: func.func @sub_broadcast
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<128x256xf32>, %[[B:.*]]: tensor<1xf32>) -> tensor<128x256xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128x256xf32>
  // CHECK: %[[OUT:.*]] = hip.sub(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<128x256xf32>, tensor<1xf32>) outs(%[[INIT]] : tensor<128x256xf32>) : tensor<128x256xf32>
  // CHECK: return %[[OUT]]
  // CHECK-NOT: hip.alloc
}
