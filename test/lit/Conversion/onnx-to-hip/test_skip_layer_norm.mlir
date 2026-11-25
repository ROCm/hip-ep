// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX SkipLayerNormalization (com.microsoft domain) is lowered to the
// fused hip.skip_layer_norm, and to a composed hip.add + hip.layer_norm
// sequence when the operands do not fit the fused op.
//
// This is STANDARD (mean-subtracting) LayerNorm with a skip/residual add and
// bias (beta) -- distinct from SkipSimplifiedLayerNormalization (RMS norm ->
// hip.skip_rms_norm).
//   sum    = input + skip [+ bias]
//   output = LayerNorm(sum, gamma, beta, epsilon)
//   output[3] (input_skip_bias_sum) = sum
//
// This test validates:
// - 4-input form (input, skip, gamma, beta) with output[3] consumed: the
//   fused op writes the sum as its second result
// - single-output form: no sum buffer
// - 5-input form (input, skip, gamma, beta, bias): bias is an operand of the
//   fused op, dynamic batch / sequence
// - a skip that broadcasts falls back to hip.add + hip.layer_norm
// - No leftover onnx.Custom
//
// MS spec reference:
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
//   #commicrosoftskiplayernormalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: 4-input / 4-output, output[0] + output[3] consumed =====

  func.func @main_graph(%input: tensor<1x16x1280xf16>,
                         %skip: tensor<1x16x1280xf16>,
                         %gamma: tensor<1280xf16>,
                         %beta: tensor<1280xf16>)
      -> (tensor<1x16x1280xf16>, tensor<1x16x1280xf16>) {
    %0:4 = "onnx.Custom"(%input, %skip, %gamma, %beta) {
      function_name = "SkipLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x16x1280xf16>, tensor<1x16x1280xf16>,
         tensor<1280xf16>, tensor<1280xf16>)
        -> (tensor<1x16x1280xf16>, none, none, tensor<1x16x1280xf16>)
    return %0#0, %0#3 : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>
  }

  // ===== Test 2: 4-input / single-output (only output[0] consumed) =====

  func.func @output_only(%input: tensor<1x16x1280xf16>,
                          %skip: tensor<1x16x1280xf16>,
                          %gamma: tensor<1280xf16>,
                          %beta: tensor<1280xf16>)
      -> tensor<1x16x1280xf16> {
    %0 = "onnx.Custom"(%input, %skip, %gamma, %beta) {
      function_name = "SkipLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x16x1280xf16>, tensor<1x16x1280xf16>,
         tensor<1280xf16>, tensor<1280xf16>)
        -> tensor<1x16x1280xf16>
    return %0 : tensor<1x16x1280xf16>
  }

  // ===== Test 3: 5-input / 4-output, bias added before the skip =====

  func.func @with_input_bias(%input: tensor<?x?x1024xf16>,
                              %skip: tensor<?x?x1024xf16>,
                              %gamma: tensor<1024xf16>,
                              %beta: tensor<1024xf16>,
                              %bias: tensor<1024xf16>)
      -> (tensor<?x?x1024xf16>, tensor<?x?x1024xf16>) {
    %0:4 = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {
      function_name = "SkipLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.00135803E-5 : f32
    } : (tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
         tensor<1024xf16>, tensor<1024xf16>, tensor<1024xf16>)
        -> (tensor<?x?x1024xf16>, none, none, tensor<?x?x1024xf16>)
    return %0#0, %0#3 : tensor<?x?x1024xf16>, tensor<?x?x1024xf16>
  }

  // ===== Test 4: skip broadcasts over the batch -> composed fallback =====

  func.func @broadcast_skip(%input: tensor<2x16x768xf32>,
                             %skip: tensor<16x768xf32>,
                             %gamma: tensor<768xf32>,
                             %beta: tensor<768xf32>)
      -> tensor<2x16x768xf32> {
    %0 = "onnx.Custom"(%input, %skip, %gamma, %beta) {
      function_name = "SkipLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<2x16x768xf32>, tensor<16x768xf32>,
         tensor<768xf32>, tensor<768xf32>)
        -> tensor<2x16x768xf32>
    return %0 : tensor<2x16x768xf32>
  }

  // ===== Test 5: broadcast skip with input bias -> (input + skip) + bias =====

  func.func @broadcast_skip_bias(%input: tensor<2x16x768xf32>,
                                 %skip: tensor<16x768xf32>,
                                 %gamma: tensor<768xf32>,
                                 %beta: tensor<768xf32>,
                                 %bias: tensor<768xf32>)
      -> tensor<2x16x768xf32> {
    %0 = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {
      function_name = "SkipLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<2x16x768xf32>, tensor<16x768xf32>,
         tensor<768xf32>, tensor<768xf32>, tensor<768xf32>)
        -> tensor<2x16x768xf32>
    return %0 : tensor<2x16x768xf32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x16x1280xf16>, %[[SKIP:.*]]: tensor<1x16x1280xf16>, %[[GAMMA:.*]]: tensor<1280xf16>, %[[BETA:.*]]: tensor<1280xf16>)
// CHECK-NOT: hip.add
// CHECK: %[[R:.*]]:2 = hip.skip_layer_norm(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]], %[[GAMMA]] : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>, tensor<1280xf16>, beta %[[BETA]] : tensor<1280xf16>) outs({{.*}}, {{.*}} : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>)
// CHECK-SAME: epsilon = 9.99999974E-6 : f32
// CHECK-NOT: hip.layer_norm
// CHECK: return %[[R]]#0, %[[R]]#1 : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @output_only
// CHECK: %[[OUT2:.*]] = hip.skip_layer_norm(%{{.*}}) ins({{.*}}) outs(%{{[^,]*}} : tensor<1x16x1280xf16>)
// CHECK: return %[[OUT2]]
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @with_input_bias
// CHECK-SAME: (%[[CTX3:.*]]: !hip.context, %[[IN3:.*]]: tensor<?x?x1024xf16>, %[[SKIP3:.*]]: tensor<?x?x1024xf16>, %[[GAMMA3:.*]]: tensor<1024xf16>, %[[BETA3:.*]]: tensor<1024xf16>, %[[BIAS3:.*]]: tensor<1024xf16>)
// CHECK-NOT: hip.add
// CHECK: %[[R3:.*]]:2 = hip.skip_layer_norm(%[[CTX3]]) ins(%[[IN3]], %[[SKIP3]], %[[GAMMA3]] : tensor<?x?x1024xf16>, tensor<?x?x1024xf16>, tensor<1024xf16>, beta %[[BETA3]] : tensor<1024xf16>, bias %[[BIAS3]] : tensor<1024xf16>)
// CHECK: return %[[R3]]#0, %[[R3]]#1 : tensor<?x?x1024xf16>, tensor<?x?x1024xf16>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @broadcast_skip
// CHECK-NOT: hip.skip_layer_norm
// CHECK: %[[SUM4:.*]] = hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<2x16x768xf32>, tensor<16x768xf32>)
// CHECK: hip.layer_norm(%{{.*}}) ins(%[[SUM4]], {{.*}})
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @broadcast_skip_bias
// CHECK-SAME: (%{{.*}}: !hip.context, %[[IN5:.*]]: tensor<2x16x768xf32>, %[[SKIP5:.*]]: tensor<16x768xf32>, %{{.*}}: tensor<768xf32>, %{{.*}}: tensor<768xf32>, %[[BIAS5:.*]]: tensor<768xf32>)
// CHECK-NOT: hip.skip_layer_norm
// CHECK: %[[SUM5:.*]] = hip.add(%{{.*}}) ins(%[[IN5]], %[[SKIP5]] : tensor<2x16x768xf32>, tensor<16x768xf32>)
// CHECK: %[[BIASED5:.*]] = hip.add(%{{.*}}) ins(%[[SUM5]], %[[BIAS5]] : tensor<2x16x768xf32>, tensor<768xf32>)
// CHECK: hip.layer_norm(%{{.*}}) ins(%[[BIASED5]], {{.*}})
// CHECK-NOT: onnx.Custom
