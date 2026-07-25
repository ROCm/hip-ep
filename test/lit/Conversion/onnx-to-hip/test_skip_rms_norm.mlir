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
// - 3-input mode (input, skip, gamma) without bias
// - 4-input mode (input, skip, gamma, bias) with optional bias
// - f16 element type support
// - 3D tensor (1x128x4096) normalization with skip connection
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// - Fused operation: input_skip_bias_sum = input + skip [+ bias],
//                    output = RMSNorm(input_skip_bias_sum) * gamma
// - 4-output ONNX pattern: (output, none, none, input_skip_bias_sum)
//
// MS spec reference:
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
//   #commicrosoftskipsimplifiedlayernormalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: Basic 3-input, 2-output =====

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

  // ===== Test 2: Dynamic shape =====

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

  // ===== Test 3: With optional bias (4 inputs) =====

  func.func @with_bias(%input: tensor<1x128x4096xf16>,
                        %skip: tensor<1x128x4096xf16>,
                        %gamma: tensor<4096xf16>,
                        %bias: tensor<4096xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {
    %output, %skip_output = "onnx.Custom"(%input, %skip, %gamma, %bias) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
         tensor<4096xf16>, tensor<4096xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
    return %output, %skip_output : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
  }

  // ===== Test 4: 4-output ONNX pattern (output, none, none, input_skip_bias_sum) =====

  func.func @four_output(%input: tensor<1x128x4096xf16>,
                          %skip: tensor<1x128x4096xf16>,
                          %gamma: tensor<4096xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {
    %0:4 = "onnx.Custom"(%input, %skip, %gamma) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> (tensor<1x128x4096xf16>, none, none, tensor<1x128x4096xf16>)
    return %0#0, %0#3 : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[SKIP:.*]]: tensor<1x128x4096xf16>, %[[GAMMA:.*]]: tensor<4096xf16>) -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]] : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>) outs({{.*}}, {{.*}} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {epsilon = 9.99999974E-6 : f32}
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @dynamic_skip_rms_norm
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<?x?xf16>, %[[SKIP:.*]]: tensor<?x?xf16>, %[[GAMMA:.*]]: tensor<?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?xf16>
// CHECK: %{{.*}} = tensor.dim %[[INPUT]], %[[C1]] : tensor<?x?xf16>
// CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
// CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]] : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>) outs({{.*}}, {{.*}} : tensor<?x?xf16>, tensor<?x?xf16>) {epsilon = 9.99999974E-6 : f32}
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @with_bias
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x128x4096xf16>, %[[SKIP:.*]]: tensor<1x128x4096xf16>, %[[GAMMA:.*]]: tensor<4096xf16>, %[[BIAS:.*]]: tensor<4096xf16>)
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]], %[[BIAS]] :
// CHECK-SAME: tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>, tensor<4096xf16>)
// CHECK-SAME: outs({{.*}}, {{.*}} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
// CHECK-SAME: {epsilon = 9.99999974E-6 : f32}
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @four_output
// CHECK: hip.skip_rms_norm(%{{.*}}) ins({{.*}}) outs({{.*}})
// CHECK-NOT: onnx.Custom
