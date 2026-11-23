// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify com.microsoft EmbedLayerNormalization lowers to Gather + Add +
// LayerNormalization, plus a sequence reduction for mask_index.
//
//   word = Gather(word_embedding, input_ids)
//   pos  = Gather(position_embedding, position_ids or 0 .. S-1)
//   sum  = word + pos [+ Gather(segment_embedding, segment_ids)]
//   Y    = LayerNorm(sum, gamma, beta)
//   mask_index = ReduceSum(mask, axis=1)     // count of ones
//   embedding_sum = sum
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // Full form: segment table, mask, and a broadcast position-id row.
  func.func @main_graph(%ids: tensor<1x4xi32>,
                       %segments: tensor<1x4xi32>,
                       %word: tensor<16x8xf16>,
                       %position: tensor<8x8xf16>,
                       %segment: tensor<2x8xf16>,
                       %gamma: tensor<8xf16>,
                       %beta: tensor<8xf16>,
                       %mask: tensor<1x4xi32>,
                       %position_ids: tensor<1x4xi32>)
      -> (tensor<1x4x8xf16>, tensor<1xi32>) {
    %0:2 = "onnx.Custom"(%ids, %segments, %word, %position, %segment,
                          %gamma, %beta, %mask, %position_ids) {
      function_name = "EmbedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.000000e-12 : f32
    } : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<16x8xf16>, tensor<8x8xf16>,
         tensor<2x8xf16>, tensor<8xf16>, tensor<8xf16>, tensor<1x4xi32>,
         tensor<1x4xi32>) -> (tensor<1x4x8xf16>, tensor<1xi32>)
    return %0#0, %0#1 : tensor<1x4x8xf16>, tensor<1xi32>
  }

  // No segment table and no mask. Position ids are omitted, so the sequence
  // positions are the static range 0 .. 3.
  func.func @default_positions(%ids: tensor<1x4xi32>,
                               %word: tensor<16x8xf16>,
                               %position: tensor<8x8xf16>,
                               %gamma: tensor<8xf16>,
                               %beta: tensor<8xf16>)
      -> (tensor<1x4x8xf16>, tensor<1xi32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %0:2 = "onnx.Custom"(%ids, %none, %word, %position, %none, %gamma, %beta) {
      function_name = "EmbedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.000000e-12 : f32
    } : (tensor<1x4xi32>, none, tensor<16x8xf16>, tensor<8x8xf16>, none,
         tensor<8xf16>, tensor<8xf16>) -> (tensor<1x4x8xf16>, tensor<1xi32>)
    return %0#0, %0#1 : tensor<1x4x8xf16>, tensor<1xi32>
  }

  // The third result is the pre-normalization sum. Batch stays dynamic, and
  // the position-id row broadcasts across it.
  func.func @embedding_sum(%ids: tensor<?x4xi32>,
                           %segments: tensor<?x4xi32>,
                           %word: tensor<16x8xf32>,
                           %position: tensor<8x8xf32>,
                           %segment: tensor<2x8xf32>,
                           %gamma: tensor<8xf32>,
                           %beta: tensor<8xf32>,
                           %mask: tensor<?x4xi32>,
                           %position_ids: tensor<1x4xi32>)
      -> (tensor<?x4x8xf32>, tensor<?xi32>, tensor<?x4x8xf32>) {
    %0:3 = "onnx.Custom"(%ids, %segments, %word, %position, %segment,
                          %gamma, %beta, %mask, %position_ids) {
      function_name = "EmbedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 1.000000e-05 : f32
    } : (tensor<?x4xi32>, tensor<?x4xi32>, tensor<16x8xf32>, tensor<8x8xf32>,
         tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x4xi32>,
         tensor<1x4xi32>)
        -> (tensor<?x4x8xf32>, tensor<?xi32>, tensor<?x4x8xf32>)
    return %0#0, %0#1, %0#2 : tensor<?x4x8xf32>, tensor<?xi32>, tensor<?x4x8xf32>
  }

  func.func @wrong_domain(%ids: tensor<1x4xi32>,
                          %word: tensor<16x8xf16>,
                          %position: tensor<8x8xf16>,
                          %gamma: tensor<8xf16>,
                          %beta: tensor<8xf16>) -> tensor<1x4x8xf16> {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Custom"(%ids, %none, %word, %position, %none, %gamma, %beta) {
      function_name = "EmbedLayerNormalization",
      domain_name = "ai.onnx"
    } : (tensor<1x4xi32>, none, tensor<16x8xf16>, tensor<8x8xf16>, none,
         tensor<8xf16>, tensor<8xf16>) -> tensor<1x4x8xf16>
    return %0 : tensor<1x4x8xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[WORD:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<16x8xf16>, tensor<1x4xi32>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[POS:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<8x8xf16>, tensor<1x4xi32>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[WP:.*]] = hip.add({{.*}}) ins(%[[WORD]], %[[POS]] : tensor<1x4x8xf16>, tensor<1x4x8xf16>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[SEG:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<2x8xf16>, tensor<1x4xi32>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[SUM:.*]] = hip.add({{.*}}) ins(%[[WP]], %[[SEG]] : tensor<1x4x8xf16>, tensor<1x4x8xf16>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[OUT:.*]] = hip.layer_norm({{.*}}) ins(%[[SUM]], {{.*}}, {{.*}} : tensor<1x4x8xf16>, tensor<8xf16>, tensor<8xf16>) outs({{.*}} : tensor<1x4x8xf16>)
// CHECK: %[[MASK:.*]] = hip.reduce_sum({{.*}}) ins({{.*}}, {{.*}} : tensor<1x4xi32>, tensor<1xi64>) outs({{.*}} : tensor<1xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 1>}
// CHECK: return %[[OUT]], %[[MASK]] : tensor<1x4x8xf16>, tensor<1xi32>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @default_positions
// CHECK: %[[IOTA:.*]] = arith.constant dense<{{\[\[0, 1, 2, 3\]\]}}> : tensor<1x4xi32>
// CHECK: hip.gather({{.*}}) ins({{.*}}, %[[IOTA]] : tensor<8x8xf16>, tensor<1x4xi32>)
// CHECK: hip.layer_norm({{.*}})
// CHECK: linalg.fill
// CHECK-NOT: hip.reduce_sum
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @embedding_sum
// CHECK: %[[WORD2:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<16x8xf32>, tensor<?x4xi32>) outs({{.*}} : tensor<?x4x8xf32>)
// CHECK: %[[POS2:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<8x8xf32>, tensor<1x4xi32>) outs({{.*}} : tensor<1x4x8xf32>)
// CHECK: %[[WP2:.*]] = hip.add({{.*}}) ins(%[[WORD2]], %[[POS2]] : tensor<?x4x8xf32>, tensor<1x4x8xf32>) outs({{.*}} : tensor<?x4x8xf32>)
// CHECK: %[[SEG2:.*]] = hip.gather({{.*}}) ins({{.*}}, {{.*}} : tensor<2x8xf32>, tensor<?x4xi32>) outs({{.*}} : tensor<?x4x8xf32>)
// CHECK: %[[SUM2:.*]] = hip.add({{.*}}) ins(%[[WP2]], %[[SEG2]] : tensor<?x4x8xf32>, tensor<?x4x8xf32>) outs({{.*}} : tensor<?x4x8xf32>)
// CHECK: %[[OUT2:.*]] = hip.layer_norm({{.*}}) ins(%[[SUM2]], {{.*}}, {{.*}} : tensor<?x4x8xf32>, tensor<8xf32>, tensor<8xf32>) outs({{.*}} : tensor<?x4x8xf32>)
// CHECK: %[[MASK2:.*]] = hip.reduce_sum({{.*}}) ins({{.*}} : tensor<?x4xi32>, tensor<1xi64>) outs({{.*}} : tensor<?xi32>) {keepdims = 0 : i64, reduced_axes = array<i64: 1>}
// CHECK: return %[[OUT2]], %[[MASK2]], %[[SUM2]] : tensor<?x4x8xf32>, tensor<?xi32>, tensor<?x4x8xf32>
// CHECK-NOT: onnx.Custom

// CHECK-LABEL: func.func @wrong_domain
// CHECK: onnx.Custom
