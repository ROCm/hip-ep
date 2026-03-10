// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Sigmoid is correctly lowered to hip.sigmoid.
//
// Assertions:
// - context argument prepended
// - input and output tensor types preserved
// - tensor.empty() used as output init (no hip.alloc)
// - hip.sigmoid result type matches input type
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // --------------------------------------------------------------------------
  // Main test case from PR17 (f16, LLaMA model size)
  // --------------------------------------------------------------------------
  func.func @main_graph(%input: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16> {
    %output = "onnx.Sigmoid"(%input) : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    return %output : tensor<1x128x14336xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
  // CHECK-SAME: %[[INPUT:.*]]: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
  // CHECK: tensor.empty() : tensor<1x128x14336xf16>
  // CHECK: hip.sigmoid(%[[CTX]]) ins(%[[INPUT]] : tensor<1x128x14336xf16>) outs({{.*}} : tensor<1x128x14336xf16>) : tensor<1x128x14336xf16>
  // CHECK-NOT: hip.alloc

  // --------------------------------------------------------------------------
  // Additional test cases
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
