// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX BiasGelu (com.microsoft domain) lowers to hip.bias_gelu.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%data: tensor<1x128x768xf16>,
                         %bias: tensor<768xf16>) -> tensor<1x128x768xf16> {
    %0 = "onnx.Custom"(%data, %bias) {
      function_name = "BiasGelu",
      domain_name = "com.microsoft"
    } : (tensor<1x128x768xf16>, tensor<768xf16>) -> tensor<1x128x768xf16>
    return %0 : tensor<1x128x768xf16>
  }

  func.func @dynamic(%data: tensor<?x?xf16>, %bias: tensor<?xf16>)
      -> tensor<?x?xf16> {
    %0 = "onnx.Custom"(%data, %bias) {
      function_name = "BiasGelu",
      domain_name = "com.microsoft"
    } : (tensor<?x?xf16>, tensor<?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }

  func.func @swapped(%bias: tensor<768xf16>, %data: tensor<1x128x768xf16>)
      -> tensor<1x128x768xf16> {
    %0 = "onnx.Custom"(%bias, %data) {
      function_name = "BiasGelu",
      domain_name = "com.microsoft"
    } : (tensor<768xf16>, tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    return %0 : tensor<1x128x768xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[DATA:.*]]: tensor<1x128x768xf16>, %[[BIAS:.*]]: tensor<768xf16>)
// CHECK: hip.bias_gelu(%[[CTX]]) ins(%[[DATA]], %[[BIAS]] : tensor<1x128x768xf16>, tensor<768xf16>) outs({{.*}} : tensor<1x128x768xf16>)
// CHECK-NOT: hip.add
// CHECK-NOT: hip.gelu
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @dynamic
// CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[DATA2:.*]]: tensor<?x?xf16>, %[[BIAS2:.*]]: tensor<?xf16>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[DATA2]], %[[C0]] : tensor<?x?xf16>
// CHECK: %[[DIM1:.*]] = tensor.dim %[[DATA2]], %[[C1]] : tensor<?x?xf16>
// CHECK: hip.bias_gelu(%[[CTX2]]) ins(%[[DATA2]], %[[BIAS2]] : tensor<?x?xf16>, tensor<?xf16>) outs(%{{.*}} : tensor<?x?xf16>)
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @swapped
// CHECK: hip.bias_gelu(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x128x768xf16>, tensor<768xf16>)
// CHECK-NOT: onnx.Custom
