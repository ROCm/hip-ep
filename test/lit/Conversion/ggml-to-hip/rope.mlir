// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/single.mlir --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s --check-prefix=LOWERED
// RUN: hip-mlir-opt %t/shared.mlir --hip-add-context-arg --convert-ggml-to-hip --cse | FileCheck %s --check-prefix=SHARED

// A rotation's angles are computed in the step from its positions, as
// llama.cpp's GPU RoPE computes them: the position times powf(theta_scale,
// pair), with theta_scale = powf(base, -2 / dimension); the amplitude is the
// attention factor. hip.rope reads the cosines and sines by row (no
// position_ids). Nothing says how far the positions reach, because nothing
// has to.
// LOWERED-LABEL: func.func @single
// LOWERED: %[[POWERS:.*]] = hip.constant {source_name = "ggml.rope.powers", value = dense<{{\[\[\[}}1.000000e+00, 0.00999999977]]]> : tensor<1x1x2xf32>}
// LOWERED: %[[POSITIONS:.*]] = hip.cast
// LOWERED-SAME: : tensor<1x2x1xf32>
// LOWERED: %[[ANGLES:.*]] = hip.mul(%{{.*}}) ins(%[[POSITIONS]], %[[POWERS]]
// LOWERED: %[[COS:.*]] = hip.cos(%{{.*}}) ins(%[[ANGLES]]
// LOWERED: %[[SIN:.*]] = hip.sin(%{{.*}}) ins(%[[ANGLES]]
// LOWERED-NOT: position_ids
// LOWERED: hip.rope(%{{.*}}) ins(%{{.*}}, %[[COS]], %[[SIN]] : tensor<1x2x4xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) outs
// LOWERED-SAME: interleaved = 1 : i64
// LOWERED-NOT: ggml.rope

// Rotations by the same positions share their constants, and so (CSE) one
// computation of the cosines and sines.
// SHARED-LABEL: func.func @shared
// SHARED-COUNT-1: hip.cos
// SHARED-NOT: hip.cos
// SHARED-COUNT-2: hip.rope

//--- single.mlir
module {
  func.func @single(%input: tensor<1x2x1x4xf32>,
                    %positions: tensor<1x1x1x2xi32>) -> tensor<1x2x1x4xf32> {
    %result = "ggml.rope"(%input, %positions) <{
      attention_factor = 1.0 : f32, beta_fast = 32.0 : f32,
      beta_slow = 1.0 : f32, dimensions = 4 : i64,
      extension_factor = 0.0 : f32, frequency_base = 10000.0 : f32,
      frequency_scale = 1.0 : f32, mode = #ggml.rope_mode<normal>,
      original_context = 8 : i64
    }> : (tensor<1x2x1x4xf32>, tensor<1x1x1x2xi32>)
      -> tensor<1x2x1x4xf32>
    return %result : tensor<1x2x1x4xf32>
  }
}

//--- shared.mlir
module {
  func.func @shared(%query: tensor<1x2x2x4xf32>, %key: tensor<1x2x1x4xf32>,
                    %positions: tensor<1x1x1x2xi32>)
      -> (tensor<1x2x2x4xf32>, tensor<1x2x1x4xf32>) {
    %q = "ggml.rope"(%query, %positions) <{
      attention_factor = 1.0 : f32, beta_fast = 32.0 : f32,
      beta_slow = 1.0 : f32, dimensions = 4 : i64,
      extension_factor = 0.0 : f32, frequency_base = 10000.0 : f32,
      frequency_scale = 1.0 : f32, mode = #ggml.rope_mode<neox>,
      original_context = 8 : i64
    }> : (tensor<1x2x2x4xf32>, tensor<1x1x1x2xi32>)
      -> tensor<1x2x2x4xf32>
    %k = "ggml.rope"(%key, %positions) <{
      attention_factor = 1.0 : f32, beta_fast = 32.0 : f32,
      beta_slow = 1.0 : f32, dimensions = 4 : i64,
      extension_factor = 0.0 : f32, frequency_base = 10000.0 : f32,
      frequency_scale = 1.0 : f32, mode = #ggml.rope_mode<neox>,
      original_context = 8 : i64
    }> : (tensor<1x2x1x4xf32>, tensor<1x1x1x2xi32>)
      -> tensor<1x2x1x4xf32>
    return %q, %k : tensor<1x2x2x4xf32>, tensor<1x2x1x4xf32>
  }
}
