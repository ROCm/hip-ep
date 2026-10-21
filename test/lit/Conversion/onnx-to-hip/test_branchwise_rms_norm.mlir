// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // Flattened layout: (B, S, C*H) is normalized as (B, S, C, H) rows with a
  // per-branch (C*H) gain.
  // CHECK-LABEL: func.func @branchwise_flattened
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x10240xf16>, %[[S:.*]]: tensor<10240xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: %[[E:.*]] = tensor.expand_shape %[[X]] {{\[\[}}0], [1], [2, 3]] {{.*}} : tensor<?x?x10240xf16> into tensor<?x?x4x2560xf16>
  // CHECK: %[[N:.*]] = hip.rms_norm(%[[CTX]]) ins(%[[E]], %[[S]] : tensor<?x?x4x2560xf16>, tensor<10240xf16>) outs({{.*}} : tensor<?x?x4x2560xf16>) {axis = -1 : i64, epsilon = 9.99999997E-7 : f32, stash_type = 1 : i64}
  // CHECK: tensor.collapse_shape %[[N]] {{\[\[}}0], [1], [2, 3]] : tensor<?x?x4x2560xf16> into tensor<?x?x10240xf16>
  func.func @branchwise_flattened(%x: tensor<?x?x10240xf16>, %s: tensor<10240xf16>) -> tensor<?x?x10240xf16> {
    %y = "onnx.Custom"(%x, %s) {
      function_name = "BranchwiseRMSNorm",
      domain_name = "com.microsoft",
      epsilon = 9.99999997E-7 : f32,
      num_branches = 4 : si64
    } : (tensor<?x?x10240xf16>, tensor<10240xf16>) -> tensor<?x?x10240xf16>
    return %y : tensor<?x?x10240xf16>
  }

  // Grouped layout (num_branches = 0) is already (..., C, H).
  // CHECK-LABEL: func.func @branchwise_grouped_shared_scale
  // CHECK-NOT: tensor.expand_shape
  // CHECK: hip.rms_norm({{.*}}) ins({{.*}} : tensor<2x4x8xf32>, tensor<8xf32>)
  // CHECK-NOT: tensor.collapse_shape
  func.func @branchwise_grouped_shared_scale(%x: tensor<2x4x8xf32>, %s: tensor<8xf32>) -> tensor<2x4x8xf32> {
    %y = "onnx.Custom"(%x, %s) {
      function_name = "BranchwiseRMSNorm",
      domain_name = "com.microsoft"
    } : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
    return %y : tensor<2x4x8xf32>
  }
}
