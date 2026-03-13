// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX SkipSimplifiedLayerNormalization (com.microsoft domain) is
// correctly lowered to hip.skip_rms_norm operation in tensor-first mode.
//
// This test validates:
// - Custom operation lowering (onnx.Custom -> hip.skip_rms_norm)
// - Domain check: only "com.microsoft" domain is converted
// - Skip RMS normalization attributes: epsilon
// - f16 element type support
// - 3D tensor (1x128x4096) normalization with skip connection
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// - Fused operation: skip_output = input + skip, output = RMSNorm(skip_output) * gamma
//
// Model: Llama-3.1-8B skip RMS layer normalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x4096xf16>,
                         %skip: tensor<1x128x4096xf16>,
                         %gamma: tensor<4096xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {
    %output, %skip_output = "onnx.Custom"(%input, %skip, %gamma) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
    return %output, %skip_output : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
  }

  // ===== Dynamic shape test =====

  func.func @dynamic_skip_rms_norm(%input: tensor<?x?xf16>,
                                    %skip: tensor<?x?xf16>,
                                    %gamma: tensor<?xf16>)
      -> (tensor<?x?xf16>, tensor<?x?xf16>) {
    %output, %skip_output = "onnx.Custom"(%input, %skip, %gamma) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.0e-05 : f32
    } : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>)
        -> (tensor<?x?xf16>, tensor<?x?xf16>)
    return %output, %skip_output : tensor<?x?xf16>, tensor<?x?xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[SKIP:.*]]: tensor<1x128x4096xf16>, %[[GAMMA:.*]]: tensor<4096xf16>) -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]] : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>) outs({{.*}}, {{.*}} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {epsilon = 9.99999974E-6 : f32} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @dynamic_skip_rms_norm
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<?x?xf16>, %[[SKIP:.*]]: tensor<?x?xf16>, %[[GAMMA:.*]]: tensor<?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?xf16>
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?xf16>
// CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
// CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]] : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>) outs({{.*}}, {{.*}} : tensor<?x?xf16>, tensor<?x?xf16>) {epsilon = 9.99999974E-6 : f32} : tensor<?x?xf16>, tensor<?x?xf16>
// CHECK-NOT: onnx.Custom
