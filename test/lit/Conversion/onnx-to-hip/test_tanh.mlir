// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Tanh is correctly lowered to hip.tanh.
//
// Assertions:
// - context argument prepended
// - input and output tensor types preserved
// - tensor.empty() used as output init (no hip.alloc)
// - hip.tanh result type matches input type
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16> {
    %output = "onnx.Tanh"(%input) : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    return %output : tensor<1x128x14336xf16>
  }

  // Dynamic shape test
  func.func @tanh_dynamic(%input: tensor<?x?x512xf32>) -> tensor<?x?x512xf32> {
    %output = "onnx.Tanh"(%input) : (tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
    return %output : tensor<?x?x512xf32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context,
// CHECK-SAME: %[[INPUT:.*]]: tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
// CHECK: tensor.empty() : tensor<1x128x14336xf16>
// CHECK: hip.tanh(%[[CTX]]) ins(%[[INPUT]] : tensor<1x128x14336xf16>) outs({{.*}} : tensor<1x128x14336xf16>) : tensor<1x128x14336xf16>
// CHECK-NOT: hip.alloc

// CHECK-LABEL: func.func @tanh_dynamic
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[IN:.*]]: tensor<?x?x512xf32>) -> tensor<?x?x512xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x512xf32>
// CHECK: hip.tanh(%[[CTX]]) ins(%[[IN]] : tensor<?x?x512xf32>) outs(%[[INIT]] : tensor<?x?x512xf32>) : tensor<?x?x512xf32>
// CHECK-NOT: hip.alloc
