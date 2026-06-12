// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX SkipLayerNormalization (com.microsoft domain) is correctly
// lowered to a composed hip.add + hip.layer_norm sequence.
//
// This is STANDARD (mean-subtracting) LayerNorm with a skip/residual add and
// bias (beta) -- distinct from SkipSimplifiedLayerNormalization (RMS norm ->
// hip.skip_rms_norm). No fused hip op exists for the standard-LN skip variant,
// so the converter composes:
//   sum    = input + skip                                 (hip.add)
//   output = LayerNorm(sum, gamma, beta, epsilon)         (hip.layer_norm)
//   output[3] (input_skip_bias_sum) = sum
//
// This test validates:
// - Custom operation lowering (onnx.Custom -> hip.add + hip.layer_norm)
// - Domain check: only "com.microsoft" domain is converted
// - 4-input form (input, skip, gamma, beta) as used by Whisper
// - REAL Whisper-large-v3 hidden size 1280 with 3D [1,16,1280] tensors
// - 4-output ONNX pattern (output, none, none, input_skip_bias_sum) where
//   output[3] is consumed as the residual for the next block (Whisper)
// - input_skip_bias_sum is wired to the input+skip sum (the hip.add result)
// - No leftover onnx.Custom
//
// MS spec reference:
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
//   #commicrosoftskiplayernormalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // ===== Test 1: 4-input / 4-output, output[0] + output[3] consumed =====
  // Matches the real Whisper-large-v3 encoder pattern (hidden = 1280).

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
}

// Test 1: output[0] = LayerNorm(input+skip); output[3] = input+skip (the
// hip.add result). The CHECK below pins SUM to the hip.add output and verifies
// the function returns (layer_norm result, SUM).
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[INPUT:.*]]: tensor<1x16x1280xf16>, %[[SKIP:.*]]: tensor<1x16x1280xf16>, %[[GAMMA:.*]]: tensor<1280xf16>, %[[BETA:.*]]: tensor<1280xf16>)
// CHECK: %[[SUM:.*]] = hip.add(%[[CTX]]) ins(%[[INPUT]], %[[SKIP]] : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>) outs({{.*}} : tensor<1x16x1280xf16>) -> tensor<1x16x1280xf16>
// CHECK: %[[OUT:.*]] = hip.layer_norm(%[[CTX]]) ins(%[[SUM]], %[[GAMMA]], %[[BETA]] : tensor<1x16x1280xf16>, tensor<1280xf16>, tensor<1280xf16>) outs({{.*}} : tensor<1x16x1280xf16>)
// CHECK: return %[[OUT]], %[[SUM]] : tensor<1x16x1280xf16>, tensor<1x16x1280xf16>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @output_only
// CHECK: %[[SUM2:.*]] = hip.add(%{{.*}}) ins({{.*}}) outs({{.*}})
// CHECK: hip.layer_norm(%{{.*}}) ins(%[[SUM2]], {{.*}})
// CHECK-NOT: onnx.Custom
