// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @scaled_silu_alpha
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x320xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: %[[S:.*]] = hip.constant {{.*}} value = dense<2.500000e-01> : tensor<f16>
  // CHECK: %[[Z:.*]] = hip.mul(%[[CTX]]) ins(%[[X]], %[[S]] : tensor<?x?x320xf16>, tensor<f16>)
  // CHECK: hip.swish(%[[CTX]]) ins(%[[Z]] : tensor<?x?x320xf16>) outs({{.*}}) : tensor<?x?x320xf16>{{$}}
  func.func @scaled_silu_alpha(%x: tensor<?x?x320xf16>) -> tensor<?x?x320xf16> {
    %y = "onnx.Custom"(%x) {
      function_name = "ScaledSiLU",
      domain_name = "com.microsoft",
      alpha = 2.500000e-01 : f32
    } : (tensor<?x?x320xf16>) -> tensor<?x?x320xf16>
    return %y : tensor<?x?x320xf16>
  }

  // alpha defaults to 1, where z = x and only the swish remains.
  // CHECK-LABEL: func.func @scaled_silu_default_alpha
  // CHECK-NOT: onnx.Custom
  // CHECK-NOT: hip.mul
  // CHECK: hip.swish
  func.func @scaled_silu_default_alpha(%x: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %y = "onnx.Custom"(%x) {
      function_name = "ScaledSiLU",
      domain_name = "com.microsoft"
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %y : tensor<2x8xf32>
  }

  // The scale input overrides alpha.
  // CHECK-LABEL: func.func @scaled_silu_scale_input
  // CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[X3:.*]]: tensor<2x8xf16>, %[[S3:.*]]: tensor<f16>)
  // CHECK-NOT: hip.constant
  // CHECK: %[[Z3:.*]] = hip.mul(%[[CTX3]]) ins(%[[X3]], %[[S3]] : tensor<2x8xf16>, tensor<f16>)
  // CHECK: hip.swish(%[[CTX3]]) ins(%[[Z3]] : tensor<2x8xf16>)
  func.func @scaled_silu_scale_input(%x: tensor<2x8xf16>, %s: tensor<f16>) -> tensor<2x8xf16> {
    %y = "onnx.Custom"(%x, %s) {
      function_name = "ScaledSiLU",
      domain_name = "com.microsoft",
      alpha = 2.500000e-01 : f32
    } : (tensor<2x8xf16>, tensor<f16>) -> tensor<2x8xf16>
    return %y : tensor<2x8xf16>
  }
}
