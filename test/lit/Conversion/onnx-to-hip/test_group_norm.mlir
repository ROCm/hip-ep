// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify com.microsoft GroupNorm lowers to hip.group_norm.
//
// 1. NCHW with SiLU (activation = 1, channels_last = 0)
// 2. NHWC, default activation omitted so it stays 0
// 3. A non-microsoft GroupNorm stays onnx.Custom

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @group_norm_nchw_silu(%X: tensor<1x8x4x4xf16>,
                                  %gamma: tensor<8xf16>,
                                  %beta: tensor<8xf16>) -> tensor<1x8x4x4xf16> {
    %Y = "onnx.Custom"(%X, %gamma, %beta)
        <{function_name = "GroupNorm"}>
        {domain_name = "com.microsoft",
         groups = 4 : si64,
         epsilon = 9.99999974E-6 : f32,
         activation = 1 : si64,
         channels_last = 0 : si64}
        : (tensor<1x8x4x4xf16>, tensor<8xf16>, tensor<8xf16>)
        -> tensor<1x8x4x4xf16>
    return %Y : tensor<1x8x4x4xf16>
  }

  // CHECK-LABEL: func.func @group_norm_nchw_silu
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x8x4x4xf16>, %[[GAMMA:.*]]: tensor<8xf16>, %[[BETA:.*]]: tensor<8xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: tensor.empty() : tensor<1x8x4x4xf16>
  // CHECK: hip.group_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[GAMMA]], %[[BETA]] :
  // CHECK-SAME: activation = 1
  // CHECK-SAME: channels_last = 0
  // CHECK-SAME: groups = 4

  func.func @group_norm_nhwc(%X: tensor<1x4x4x8xf32>,
                             %gamma: tensor<8xf32>,
                             %beta: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    %Y = "onnx.Custom"(%X, %gamma, %beta)
        <{function_name = "GroupNorm"}>
        {domain_name = "com.microsoft",
         groups = 2 : si64,
         activation = 0 : si64,
         channels_last = 1 : si64}
        : (tensor<1x4x4x8xf32>, tensor<8xf32>, tensor<8xf32>)
        -> tensor<1x4x4x8xf32>
    return %Y : tensor<1x4x4x8xf32>
  }

  // CHECK-LABEL: func.func @group_norm_nhwc
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.group_norm
  // CHECK-SAME: groups = 2

  func.func @group_norm_wrong_domain(%X: tensor<1x8x4x4xf32>,
                                     %gamma: tensor<8xf32>,
                                     %beta: tensor<8xf32>) -> tensor<1x8x4x4xf32> {
    %Y = "onnx.Custom"(%X, %gamma, %beta)
        <{function_name = "GroupNorm"}>
        {domain_name = "com.example",
         groups = 4 : si64,
         activation = 0 : si64,
         channels_last = 0 : si64}
        : (tensor<1x8x4x4xf32>, tensor<8xf32>, tensor<8xf32>)
        -> tensor<1x8x4x4xf32>
    return %Y : tensor<1x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @group_norm_wrong_domain
  // CHECK: onnx.Custom
  // CHECK-NOT: hip.group_norm
}
