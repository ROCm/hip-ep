// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // CHECK-LABEL: func.func @engram_gate_two_outputs
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[K:.*]]: tensor<?x?x4x2560xf16>, %[[Q:.*]]: tensor<?x?x4x2560xf16>, %[[V:.*]]: tensor<?x?x2560xf16>, %[[KS:.*]]: tensor<4x2560xf16>, %[[QS:.*]]: tensor<4x2560xf16>, %[[CS:.*]]: tensor<4x2560xf16>)
  // CHECK-NOT: onnx.Custom
  // CHECK: %[[R:.*]]:2 = hip.engram_gate(%[[CTX]]) ins(%[[K]], %[[Q]], %[[V]], %[[KS]], %[[QS]], %[[CS]] : {{.*}}) outs({{.*}}, {{.*}} : tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>) {epsilon = 9.99999997E-7 : f32}
  // CHECK: return %[[R]]#0, %[[R]]#1
  func.func @engram_gate_two_outputs(%k: tensor<?x?x4x2560xf16>, %q: tensor<?x?x4x2560xf16>, %v: tensor<?x?x2560xf16>, %ks: tensor<4x2560xf16>, %qs: tensor<4x2560xf16>, %cs: tensor<4x2560xf16>) -> (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>) {
    %y:2 = "onnx.Custom"(%k, %q, %v, %ks, %qs, %cs) {
      function_name = "EngramGate",
      domain_name = "com.microsoft",
      epsilon = 9.99999997E-7 : f32
    } : (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>, tensor<?x?x2560xf16>, tensor<4x2560xf16>, tensor<4x2560xf16>, tensor<4x2560xf16>) -> (tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>)
    return %y#0, %y#1 : tensor<?x?x4x2560xf16>, tensor<?x?x4x2560xf16>
  }

  // CHECK-LABEL: func.func @engram_gate_one_output
  // CHECK: hip.engram_gate({{.*}}) ins({{[^)]*}} : tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>, tensor<1x2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>) outs({{[^,]*}} : tensor<1x2x2x8xf32>)
  func.func @engram_gate_one_output(%k: tensor<1x2x2x8xf32>, %q: tensor<1x2x2x8xf32>, %v: tensor<1x2x8xf32>, %ks: tensor<2x8xf32>, %qs: tensor<2x8xf32>) -> tensor<1x2x2x8xf32> {
    %y = "onnx.Custom"(%k, %q, %v, %ks, %qs) {
      function_name = "EngramGate",
      domain_name = "com.microsoft"
    } : (tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>, tensor<1x2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<1x2x2x8xf32>
    return %y : tensor<1x2x2x8xf32>
  }
}
