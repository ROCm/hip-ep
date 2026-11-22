// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @gated_add
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x2560xf16>, %[[Y:.*]]: tensor<?x?x2560xf16>, %[[G:.*]]: tensor<?x?x1xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: %[[P:.*]] = hip.mul(%[[CTX]]) ins(%[[Y]], %[[G]] : tensor<?x?x2560xf16>, tensor<?x?x1xf16>)
  // CHECK: hip.add(%[[CTX]]) ins(%[[X]], %[[P]] : tensor<?x?x2560xf16>, tensor<?x?x2560xf16>)
  func.func @gated_add(%x: tensor<?x?x2560xf16>, %y: tensor<?x?x2560xf16>, %g: tensor<?x?x1xf16>) -> tensor<?x?x2560xf16> {
    %out = "onnx.Custom"(%x, %y, %g) {
      function_name = "GatedAdd",
      domain_name = "com.microsoft"
    } : (tensor<?x?x2560xf16>, tensor<?x?x2560xf16>, tensor<?x?x1xf16>) -> tensor<?x?x2560xf16>
    return %out : tensor<?x?x2560xf16>
  }

  // The gate broadcasts only along the last axis.
  // CHECK-LABEL: func.func @gated_add_full_gate
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.add
  func.func @gated_add_full_gate(%x: tensor<2x8xf32>, %y: tensor<2x8xf32>, %g: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %out = "onnx.Custom"(%x, %y, %g) {
      function_name = "GatedAdd",
      domain_name = "com.microsoft"
    } : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    return %out : tensor<2x8xf32>
  }
}
