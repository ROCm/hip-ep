// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX SimplifiedLayerNormalization is correctly lowered to
// hip.rms_norm operation in tensor-first mode.
//
// This test validates:
// - Custom operation lowering (onnx.Custom -> hip.rms_norm)
// - RMS normalization attributes: epsilon, axis, stash_type
// - f16 element type support
// - 3D tensor (1x128x4096) normalization
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
//
// Model: Llama-3.1-8B RMS layer normalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x4096xf16>,
                         %scale: tensor<4096xf16>)
      -> tensor<1x128x4096xf16> {
    %output = "onnx.Custom"(%input, %scale) {
      function_name = "SimplifiedLayerNormalization",
      epsilon = 9.99999974E-6 : f32,
      axis = -1 : si64,
      stash_type = 1 : si64
    } : (tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> tensor<1x128x4096xf16>
    return %output : tensor<1x128x4096xf16>
  }

  // ===== Dynamic shape test =====

  func.func @dynamic_rms_norm(%input: tensor<?x?xf16>,
                               %scale: tensor<?xf16>)
      -> tensor<?x?xf16> {
    %output = "onnx.Custom"(%input, %scale) {
      function_name = "SimplifiedLayerNormalization",
      epsilon = 1.0e-05 : f32,
      axis = -1 : si64,
      stash_type = 1 : si64
    } : (tensor<?x?xf16>, tensor<?xf16>)
        -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context,
// CHECK-SAME: %[[INPUT:.*]]: tensor<1x128x4096xf16>,
// CHECK-SAME: %[[SCALE:.*]]: tensor<4096xf16>) -> tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: hip.rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SCALE]] : tensor<1x128x4096xf16>, tensor<4096xf16>) outs({{.*}} : tensor<1x128x4096xf16>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @dynamic_rms_norm
// CHECK-SAME: (%[[CTX:.*]]: !hip.context,
// CHECK-SAME: %[[INPUT:.*]]: tensor<?x?xf16>,
// CHECK-SAME: %[[SCALE:.*]]: tensor<?xf16>) -> tensor<?x?xf16>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?xf16>
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?xf16>
// CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
// CHECK: hip.rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SCALE]] : tensor<?x?xf16>, tensor<?xf16>) outs({{.*}} : tensor<?x?xf16>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
// CHECK-NOT: onnx.Custom
