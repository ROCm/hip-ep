// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// An elementwise inference batch normalization over (N, C, H, W) with C = 4:
//   y = (x - mean) * (1 / sqrt(variance + epsilon)) * scale + bias
// Each per-channel vector reaches the elementwise ops through two chained
// expand_shape views, which compose into one before the chain is matched. The
// parameters are function arguments: the rewrite does not need constants.
// epsilon = 1e-5 is the default of hip.batch_norm, so it is not printed.

// CHECK-LABEL: func.func @batch_norm_two_views
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK-NOT:     hip.sub
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<1x4x2x2xf32>
// CHECK-NEXT:    %[[BN:.*]] = hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] : tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
// CHECK-NOT:     hip.
// CHECK:         return %[[BN]]
func.func @batch_norm_two_views(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>,
                                %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                %scale: tensor<4xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x2x2xf32> {
  %epsilon_f64 = arith.constant 1.000000e-05 : f64
  %epsilon_f32 = arith.truncf %epsilon_f64 : f64 to f32
  %epsilon = tensor.splat %epsilon_f32 : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_2d = tensor.expand_shape %mean [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %mean_view = tensor.expand_shape %mean_2d [[0], [1, 2]] output_shape [4, 1, 1]
      : tensor<4x1xf32> into tensor<4x1x1xf32>
  %inverse_2d = tensor.expand_shape %inverse [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %inverse_view = tensor.expand_shape %inverse_2d [[0], [1, 2]] output_shape [4, 1, 1]
      : tensor<4x1xf32> into tensor<4x1x1xf32>
  %scale_2d = tensor.expand_shape %scale [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %scale_view = tensor.expand_shape %scale_2d [[0], [1, 2]] output_shape [4, 1, 1]
      : tensor<4x1xf32> into tensor<4x1x1xf32>
  %bias_2d = tensor.expand_shape %bias [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %bias_view = tensor.expand_shape %bias_2d [[0], [1, 2]] output_shape [4, 1, 1]
      : tensor<4x1xf32> into tensor<4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted : tensor<1x4x2x2xf32>
}

// -----

// One view per vector, aligned with the input's rank ([1, C, 1, 1]), constant
// parameters, and an epsilon that is a splat hip.constant.

// CHECK-LABEL: func.func @batch_norm_one_view_constants
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>)
// CHECK-DAG:     %[[MEAN:.*]] = hip.constant {value = dense<1.000000e-01> : tensor<4xf32>}
// CHECK-DAG:     %[[VAR:.*]] = hip.constant {value = dense<2.000000e+00> : tensor<4xf32>}
// CHECK-DAG:     %[[SCALE:.*]] = hip.constant {value = dense<3.000000e+00> : tensor<4xf32>}
// CHECK-DAG:     %[[BIAS:.*]] = hip.constant {value = dense<4.000000e+00> : tensor<4xf32>}
// CHECK:         %[[BN:.*]] = hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] : {{.*}}) {epsilon = 1.000000e-03 : f32}
// CHECK-NOT:     hip.sub
// CHECK-NOT:     hip.mul
// CHECK:         return %[[BN]]
func.func @batch_norm_one_view_constants(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>)
    -> tensor<1x4x2x2xf32> {
  %mean = hip.constant {value = dense<0.1> : tensor<4xf32>} : tensor<4xf32>
  %variance = hip.constant {value = dense<2.0> : tensor<4xf32>} : tensor<4xf32>
  %scale = hip.constant {value = dense<3.0> : tensor<4xf32>} : tensor<4xf32>
  %bias = hip.constant {value = dense<4.0> : tensor<4xf32>} : tensor<4xf32>
  %epsilon = hip.constant {value = dense<1.0e-3> : tensor<f32>} : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2, 3]] output_shape [1, 4, 1, 1]
      : tensor<4xf32> into tensor<1x4x1x1xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2, 3]] output_shape [1, 4, 1, 1]
      : tensor<4xf32> into tensor<1x4x1x1xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2, 3]] output_shape [1, 4, 1, 1]
      : tensor<4xf32> into tensor<1x4x1x1xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2, 3]] output_shape [1, 4, 1, 1]
      : tensor<4xf32> into tensor<1x4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<1x4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted : tensor<1x4x2x2xf32>
}

// -----

// In f16 the graph adds the epsilon after rounding it to f16, so the
// batch norm's epsilon is that rounded value (168 * 2^-24), not 1e-5.

// CHECK-LABEL: func.func @batch_norm_f16_epsilon
// CHECK:         hip.batch_norm
// CHECK-SAME:    {epsilon = 1.00135803E-5 : f32}
func.func @batch_norm_f16_epsilon(%ctx: !hip.context, %x: tensor<1x4x2x2xf16>,
                                  %mean: tensor<4xf16>, %variance: tensor<4xf16>,
                                  %scale: tensor<4xf16>, %bias: tensor<4xf16>)
    -> tensor<1x4x2x2xf16> {
  %epsilon_f64 = arith.constant 1.000000e-05 : f64
  %epsilon_f16 = arith.truncf %epsilon_f64 : f64 to f16
  %epsilon = tensor.splat %epsilon_f16 : tensor<f16>

  %e0 = tensor.empty() : tensor<4xf16>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf16>, tensor<f16>)
      outs(%e0 : tensor<4xf16>) -> tensor<4xf16>
  %e1 = tensor.empty() : tensor<4xf16>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf16>)
      outs(%e1 : tensor<4xf16>) : tensor<4xf16>
  %e2 = tensor.empty() : tensor<4xf16>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf16>)
      outs(%e2 : tensor<4xf16>) : tensor<4xf16>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf16> into tensor<4x1x1xf16>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf16> into tensor<4x1x1xf16>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf16> into tensor<4x1x1xf16>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf16> into tensor<4x1x1xf16>

  %e3 = tensor.empty() : tensor<1x4x2x2xf16>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>)
      outs(%e3 : tensor<1x4x2x2xf16>) : tensor<1x4x2x2xf16>
  %e4 = tensor.empty() : tensor<1x4x2x2xf16>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>)
      outs(%e4 : tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %e5 = tensor.empty() : tensor<1x4x2x2xf16>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>)
      outs(%e5 : tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  %e6 = tensor.empty() : tensor<1x4x2x2xf16>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>)
      outs(%e6 : tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  return %shifted : tensor<1x4x2x2xf16>
}

// -----

// The views put the vectors on the last dimension ([1, 1, C] against
// (N, C, H, W) with H = W = C), so this is a per-column affine map, not a
// batch normalization.

// CHECK-LABEL: func.func @no_fusion_view_not_on_channel
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
func.func @no_fusion_view_not_on_channel(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>,
                                         %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                         %scale: tensor<4xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x4x4xf32> {
  %epsilon = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [1, 1, 4]
      : tensor<4xf32> into tensor<1x1x4xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [1, 1, 4]
      : tensor<4xf32> into tensor<1x1x4xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [1, 1, 4]
      : tensor<4xf32> into tensor<1x1x4xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [1, 1, 4]
      : tensor<4xf32> into tensor<1x1x4xf32>

  %e3 = tensor.empty() : tensor<1x4x4x4xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>)
      outs(%e3 : tensor<1x4x4x4xf32>) : tensor<1x4x4x4xf32>
  %e4 = tensor.empty() : tensor<1x4x4x4xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>)
      outs(%e4 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %e5 = tensor.empty() : tensor<1x4x4x4xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>)
      outs(%e5 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %e6 = tensor.empty() : tensor<1x4x4x4xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>)
      outs(%e6 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %shifted : tensor<1x4x4x4xf32>
}

// -----

// The centered input is also returned, so the chain must stay to produce it.

// CHECK-LABEL: func.func @no_fusion_intermediate_has_other_use
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
func.func @no_fusion_intermediate_has_other_use(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>,
                                                %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                                %scale: tensor<4xf32>, %bias: tensor<4xf32>)
    -> (tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>) {
  %epsilon = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted, %centered : tensor<1x4x2x2xf32>, tensor<1x4x2x2xf32>
}

// -----

// mean - x is the negated centering, not the batch normalization's x - mean.

// CHECK-LABEL: func.func @no_fusion_reversed_subtraction
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
func.func @no_fusion_reversed_subtraction(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>,
                                          %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                          %scale: tensor<4xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x2x2xf32> {
  %epsilon = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%mean_view, %x : tensor<4x1x1xf32>, tensor<1x4x2x2xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted : tensor<1x4x2x2xf32>
}

// -----

// hip.batch_norm carries epsilon as an attribute, so an epsilon that is only
// known at run time leaves the chain in place.

// CHECK-LABEL: func.func @no_fusion_runtime_epsilon
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
func.func @no_fusion_runtime_epsilon(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>,
                                     %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                     %scale: tensor<4xf32>, %bias: tensor<4xf32>,
                                     %epsilon: tensor<f32>)
    -> tensor<1x4x2x2xf32> {
  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted : tensor<1x4x2x2xf32>
}

// -----

// Only the default rounding of arith.truncf is reproduced when reading
// epsilon, so a cast with an explicit rounding mode leaves the chain in place.

// CHECK-LABEL: func.func @no_fusion_epsilon_rounding_mode
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
func.func @no_fusion_epsilon_rounding_mode(%ctx: !hip.context, %x: tensor<1x4x2x2xf32>,
                                           %mean: tensor<4xf32>, %variance: tensor<4xf32>,
                                           %scale: tensor<4xf32>, %bias: tensor<4xf32>)
    -> tensor<1x4x2x2xf32> {
  %epsilon_f64 = arith.constant 1.000000e-05 : f64
  %epsilon_f32 = arith.truncf %epsilon_f64 to_nearest_away : f64 to f32
  %epsilon = tensor.splat %epsilon_f32 : tensor<f32>

  %e0 = tensor.empty() : tensor<4xf32>
  %variance_plus_epsilon = hip.add(%ctx)
      ins(%variance, %epsilon : tensor<4xf32>, tensor<f32>)
      outs(%e0 : tensor<4xf32>) -> tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %standard_deviation = hip.sqrt(%ctx) ins(%variance_plus_epsilon : tensor<4xf32>)
      outs(%e1 : tensor<4xf32>) : tensor<4xf32>
  %e2 = tensor.empty() : tensor<4xf32>
  %inverse = hip.reciprocal(%ctx) ins(%standard_deviation : tensor<4xf32>)
      outs(%e2 : tensor<4xf32>) : tensor<4xf32>

  %mean_view = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %inverse_view = tensor.expand_shape %inverse [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %scale_view = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>
  %bias_view = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1]
      : tensor<4xf32> into tensor<4x1x1xf32>

  %e3 = tensor.empty() : tensor<1x4x2x2xf32>
  %centered = hip.sub(%ctx) ins(%x, %mean_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e3 : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
  %e4 = tensor.empty() : tensor<1x4x2x2xf32>
  %normalized = hip.mul(%ctx) ins(%centered, %inverse_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e4 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e5 = tensor.empty() : tensor<1x4x2x2xf32>
  %scaled = hip.mul(%ctx) ins(%normalized, %scale_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e5 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  %e6 = tensor.empty() : tensor<1x4x2x2xf32>
  %shifted = hip.add(%ctx) ins(%scaled, %bias_view : tensor<1x4x2x2xf32>, tensor<4x1x1xf32>)
      outs(%e6 : tensor<1x4x2x2xf32>) -> tensor<1x4x2x2xf32>
  return %shifted : tensor<1x4x2x2xf32>
}

// -----

// Chained expand_shape views compose into one view of the original tensor.

// CHECK-LABEL: func.func @compose_expand_shapes
// CHECK-SAME:  (%[[V:.*]]: tensor<4xf32>)
// CHECK-NEXT:    %[[VIEW:.*]] = tensor.expand_shape %[[V]] {{\[}}[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf32> into tensor<4x1x1xf32>
// CHECK-NEXT:    return %[[VIEW]]
func.func @compose_expand_shapes(%v: tensor<4xf32>) -> tensor<4x1x1xf32> {
  %v_2d = tensor.expand_shape %v [[0, 1]] output_shape [4, 1]
      : tensor<4xf32> into tensor<4x1xf32>
  %v_3d = tensor.expand_shape %v_2d [[0], [1, 2]] output_shape [4, 1, 1]
      : tensor<4x1xf32> into tensor<4x1x1xf32>
  return %v_3d : tensor<4x1x1xf32>
}
