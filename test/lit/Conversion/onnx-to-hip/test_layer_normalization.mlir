// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX LayerNormalization (opset 17+) is correctly lowered to
// hip.layer_norm. The op is native (not decomposed).
//
// Test cases:
// 1. Basic: X + Scale + Bias → Y only (inference, static shape)
// 2. No bias: X + Scale → Y only
// 3. Dynamic shape: f16, ?x?x512
// 4. With optional Mean/InvStdDev outputs
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // --- Case 1: basic with bias (inference) ---
  func.func @layer_norm_basic(%X: tensor<2x3x4xf32>, %Scale: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<2x3x4xf32> {
    %Y = "onnx.LayerNormalization"(%X, %Scale, %B) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %Y : tensor<2x3x4xf32>
  }

  // CHECK-LABEL: func.func @layer_norm_basic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<2x3x4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[B:.*]]: tensor<4xf32>)
  // CHECK-NOT: onnx.LayerNormalization
  // CHECK: tensor.empty() : tensor<2x3x4xf32>
  // CHECK: hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[SCALE]], %[[B]] :
  // CHECK-SAME: outs(

  // --- Case 2: no bias ---
  func.func @layer_norm_no_bias(%X: tensor<1x128x4096xf16>, %Scale: tensor<4096xf16>) -> tensor<1x128x4096xf16> {
    %Y = "onnx.LayerNormalization"(%X, %Scale) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x128x4096xf16>, tensor<4096xf16>) -> tensor<1x128x4096xf16>
    return %Y : tensor<1x128x4096xf16>
  }

  // CHECK-LABEL: func.func @layer_norm_no_bias
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x4096xf16>, %[[SCALE:.*]]: tensor<4096xf16>)
  // CHECK-NOT: onnx.LayerNormalization
  // CHECK: hip.layer_norm(%[[CTX]])
  // CHECK-SAME: ins(%[[X]], %[[SCALE]] :

  // --- Case 3: dynamic shape ---
  func.func @layer_norm_dynamic(%X: tensor<?x?x512xf16>, %Scale: tensor<512xf16>, %B: tensor<512xf16>) -> tensor<?x?x512xf16> {
    %Y = "onnx.LayerNormalization"(%X, %Scale, %B) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<?x?x512xf16>, tensor<512xf16>, tensor<512xf16>) -> tensor<?x?x512xf16>
    return %Y : tensor<?x?x512xf16>
  }

  // CHECK-LABEL: func.func @layer_norm_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x512xf16>,
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[X]], %[[C0]]
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[X]], %[[C1]]
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?x512xf16>
  // CHECK: hip.layer_norm(%[[CTX]])
  // CHECK-SAME: outs(%[[INIT]] :

  // --- Case 4: all optional outputs, axis=-1 ---
  func.func @layer_norm_stats(
      %X: tensor<2x3x4xf16>, %Scale: tensor<4xf16>)
      -> (tensor<2x3x4xf16>, tensor<2x3x1xf32>, tensor<2x3x1xf32>) {
    %Y:3 = "onnx.LayerNormalization"(%X, %Scale)
      {axis = -1 : si64, epsilon = 9.99999974E-6 : f32,
       stash_type = 1 : si64}
      : (tensor<2x3x4xf16>, tensor<4xf16>)
        -> (tensor<2x3x4xf16>, tensor<2x3x1xf32>, tensor<2x3x1xf32>)
    return %Y#0, %Y#1, %Y#2
      : tensor<2x3x4xf16>, tensor<2x3x1xf32>, tensor<2x3x1xf32>
  }

  // CHECK-LABEL: func.func @layer_norm_stats
  // CHECK-DAG: %[[YINIT:.*]] = tensor.empty() : tensor<2x3x4xf16>
  // CHECK-DAG: %[[MINIT:.*]] = tensor.empty() : tensor<2x3x1xf32>
  // CHECK-DAG: %[[IINIT:.*]] = tensor.empty() : tensor<2x3x1xf32>
  // CHECK: %[[LN:.*]]:3 = hip.layer_norm
  // CHECK-SAME: outs(%[[YINIT]], %[[MINIT]], %[[IINIT]]

  // --- Case 5: dynamic prefix, positive axis ---
  func.func @layer_norm_dynamic_stats(
      %X: tensor<?x3x4xf16>, %Scale: tensor<3x4xf16>)
      -> (tensor<?x3x4xf16>, tensor<?x1x1xf32>, tensor<?x1x1xf32>) {
    %Y:3 = "onnx.LayerNormalization"(%X, %Scale)
      {axis = 1 : si64, epsilon = 9.99999974E-6 : f32,
       stash_type = 1 : si64}
      : (tensor<?x3x4xf16>, tensor<3x4xf16>)
        -> (tensor<?x3x4xf16>, tensor<?x1x1xf32>, tensor<?x1x1xf32>)
    return %Y#0, %Y#1, %Y#2
      : tensor<?x3x4xf16>, tensor<?x1x1xf32>, tensor<?x1x1xf32>
  }

  // CHECK-LABEL: func.func @layer_norm_dynamic_stats
  // CHECK-SAME: %[[DX:[^,]+]]: tensor<?x3x4xf16>
  // CHECK: %[[DN:.*]] = tensor.dim %[[DX]], %{{.*}}
  // CHECK: %[[SDN:.*]] = tensor.dim %[[DX]], %{{.*}}
  // CHECK-DAG: %[[DY:.*]] = tensor.empty(%[[DN]]) : tensor<?x3x4xf16>
  // CHECK-DAG: %[[DM:.*]] = tensor.empty(%[[SDN]]) : tensor<?x1x1xf32>
  // CHECK-DAG: %[[DI:.*]] = tensor.empty(%[[SDN]]) : tensor<?x1x1xf32>
  // CHECK: hip.layer_norm
  // CHECK-SAME: outs(%[[DY]], %[[DM]], %[[DI]]

  // --- Case 6: InvStdDev requested without Mean. HIP gets a mean scratch. ---
  func.func @layer_norm_inv_without_mean(
      %X: tensor<2x4xf16>, %Scale: tensor<4xf16>)
      -> (tensor<2x4xf16>, tensor<2x1xf32>) {
    %Y:3 = "onnx.LayerNormalization"(%X, %Scale)
      {axis = -1 : si64, epsilon = 9.99999974E-6 : f32,
       stash_type = 1 : si64}
      : (tensor<2x4xf16>, tensor<4xf16>)
        -> (tensor<2x4xf16>, none, tensor<2x1xf32>)
    return %Y#0, %Y#2 : tensor<2x4xf16>, tensor<2x1xf32>
  }

  // CHECK-LABEL: func.func @layer_norm_inv_without_mean
  // CHECK-DAG: %[[SY:.*]] = tensor.empty() : tensor<2x4xf16>
  // CHECK-DAG: %[[SM:.*]] = tensor.empty() : tensor<2x1xf32>
  // CHECK-DAG: %[[SI:.*]] = tensor.empty() : tensor<2x1xf32>
  // CHECK: %[[SLN:.*]]:3 = hip.layer_norm
  // CHECK-SAME: outs(%[[SY]], %[[SM]], %[[SI]]
}
