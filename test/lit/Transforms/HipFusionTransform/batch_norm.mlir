// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform %s | FileCheck %s

// An elementwise inference batch normalization over (N, C, H, W) with C = 4:
//   y = (x - mean) * (1 / sqrt(variance + epsilon)) * scale + bias
// Each per-channel vector is expanded to [C, 1, 1] before it meets the
// full-size tensor. The parameters are function arguments: the rewrite does not
// need constants. Cases without scale and bias use the shortest chain that shows
// the point.

!vec = tensor<4xf32>
!view = tensor<4x1x1xf32>
!full = tensor<1x4x2x2xf32>

// The form torch-mlir emits: epsilon is a truncated f64 and a view takes two
// expand_shape steps. epsilon = 1e-5 is the default of hip.batch_norm, so it is
// not printed.

// CHECK-LABEL: func.func @batch_norm_torch_form
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK-NOT:     hip.sub
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<1x4x2x2xf32>
// CHECK-NEXT:    %[[BN:.*]] = hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] : tensor<1x4x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<1x4x2x2xf32>) : tensor<1x4x2x2xf32>
// CHECK-NOT:     hip.
// CHECK:         return %[[BN]]
func.func @batch_norm_torch_form(%ctx: !hip.context, %x: !full, %mean: !vec,
                                 %var: !vec, %scale: !vec, %bias: !vec) -> !full {
  %eps_f64 = arith.constant 1.000000e-05 : f64
  %eps_f32 = arith.truncf %eps_f64 : f64 to f32
  %eps = tensor.splat %eps_f32 : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_2d = tensor.expand_shape %mean [[0, 1]] output_shape [4, 1] : !vec into tensor<4x1xf32>
  %mean_v = tensor.expand_shape %mean_2d [[0], [1, 2]] output_shape [4, 1, 1] : tensor<4x1xf32> into !view
  %inv_2d = tensor.expand_shape %inv [[0, 1]] output_shape [4, 1] : !vec into tensor<4x1xf32>
  %inv_v = tensor.expand_shape %inv_2d [[0], [1, 2]] output_shape [4, 1, 1] : tensor<4x1xf32> into !view
  %scale_v = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %bias_v = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %s = hip.mul(%ctx) ins(%n, %scale_v : !full, !view) outs(%e5 : !full) -> !full
  %e6 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%s, %bias_v : !full, !view) outs(%e6 : !full) -> !full
  return %y : !full
}

// Every multiply and add commutes: the view comes first in each op, and
// epsilon first in the variance sum.

// CHECK-LABEL: func.func @batch_norm_commuted_operands
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_commuted_operands(%ctx: !hip.context, %x: !full, %mean: !vec,
                                        %var: !vec, %scale: !vec, %bias: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%eps, %var : tensor<f32>, !vec) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %scale_v = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %bias_v = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%inv_v, %c : !view, !full) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %s = hip.mul(%ctx) ins(%scale_v, %n : !view, !full) outs(%e5 : !full) -> !full
  %e6 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%bias_v, %s : !view, !full) outs(%e6 : !full) -> !full
  return %y : !full
}

// The scale may be applied before the normalization.

// CHECK-LABEL: func.func @batch_norm_scale_before_normalization
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_scale_before_normalization(
    %ctx: !hip.context, %x: !full, %mean: !vec, %var: !vec, %scale: !vec,
    %bias: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %scale_v = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %bias_v = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %s = hip.mul(%ctx) ins(%c, %scale_v : !full, !view) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%s, %inv_v : !full, !view) outs(%e5 : !full) -> !full
  %e6 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%n, %bias_v : !full, !view) outs(%e6 : !full) -> !full
  return %y : !full
}

// Without scale and bias, as for BatchNorm(affine=False): hip.batch_norm always
// takes both, so they become constants of ones and zeros.

// CHECK-LABEL: func.func @batch_norm_without_scale_and_bias
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>)
// CHECK:         %[[ONES:.*]] = hip.constant {{.*}}dense<1.000000e+00> : tensor<4xf32>
// CHECK:         %[[ZEROS:.*]] = hip.constant {{.*}}dense<0.000000e+00> : tensor<4xf32>
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[ONES]], %[[ZEROS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_without_scale_and_bias(%ctx: !hip.context, %x: !full,
                                             %mean: !vec, %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y : !full
}

// With a scale but no bias, the bias becomes a constant of zeros.

// CHECK-LABEL: func.func @batch_norm_without_bias
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>)
// CHECK:         %[[ZEROS:.*]] = hip.constant {{.*}}dense<0.000000e+00> : tensor<4xf32>
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[ZEROS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_without_bias(%ctx: !hip.context, %x: !full, %mean: !vec,
                                   %var: !vec, %scale: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %scale_v = tensor.expand_shape %scale [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%n, %scale_v : !full, !view) outs(%e5 : !full) -> !full
  return %y : !full
}

// With a bias but no scale, the scale becomes a constant of ones.

// CHECK-LABEL: func.func @batch_norm_without_scale
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK:         %[[ONES:.*]] = hip.constant {{.*}}dense<1.000000e+00> : tensor<4xf32>
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[ONES]], %[[BIAS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_without_scale(%ctx: !hip.context, %x: !full, %mean: !vec,
                                    %var: !vec, %bias: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %bias_v = tensor.expand_shape %bias [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%n, %bias_v : !full, !view) outs(%e5 : !full) -> !full
  return %y : !full
}

// A view may have the full rank, [1, C, 1, 1], instead of [C, 1, 1].

// CHECK-LABEL: func.func @batch_norm_full_rank_views
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x4x2x2xf32>, %[[MEAN:.*]]: tensor<4xf32>, %[[VAR:.*]]: tensor<4xf32>, %[[SCALE:.*]]: tensor<4xf32>, %[[BIAS:.*]]: tensor<4xf32>)
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], %[[SCALE]], %[[BIAS]], %[[MEAN]], %[[VAR]] :
// CHECK-NOT:     hip.{{sub|mul|add}}
// CHECK:         return
func.func @batch_norm_full_rank_views(%ctx: !hip.context, %x: !full, %mean: !vec,
                                      %var: !vec, %scale: !vec, %bias: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : !vec into tensor<1x4x1x1xf32>
  %inv_v = tensor.expand_shape %inv [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : !vec into tensor<1x4x1x1xf32>
  %scale_v = tensor.expand_shape %scale [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : !vec into tensor<1x4x1x1xf32>
  %bias_v = tensor.expand_shape %bias [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : !vec into tensor<1x4x1x1xf32>

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, tensor<1x4x1x1xf32>) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%c, %inv_v : !full, tensor<1x4x1x1xf32>) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %s = hip.mul(%ctx) ins(%n, %scale_v : !full, tensor<1x4x1x1xf32>) outs(%e5 : !full) -> !full
  %e6 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%s, %bias_v : !full, tensor<1x4x1x1xf32>) outs(%e6 : !full) -> !full
  return %y : !full
}

// The normalization may divide by sqrt(variance + epsilon). A frontend that
// does not rewrite division may emit it.

// CHECK-LABEL: func.func @batch_norm_divide
// CHECK:         hip.batch_norm
// CHECK-NOT:     hip.{{sub|div|sqrt}}
// CHECK:         return
func.func @batch_norm_divide(%ctx: !hip.context, %x: !full, %mean: !vec,
                             %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %sd_v = tensor.expand_shape %sd [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e2 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e2 : !full) : !full
  %e3 = tensor.empty() : !full
  %y = hip.div(%ctx) ins(%c, %sd_v : !full, !view) outs(%e3 : !full) : !full
  return %y : !full
}

// The inverse standard deviation may be pow(variance + epsilon, -0.5).

// CHECK-LABEL: func.func @batch_norm_pow_half
// CHECK:         hip.batch_norm
// CHECK-NOT:     hip.{{sub|mul|pow}}
// CHECK:         return
func.func @batch_norm_pow_half(%ctx: !hip.context, %x: !full, %mean: !vec,
                               %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %inv = hip.pow(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) {exponent = -5.000000e-01 : f64} : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e2 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e2 : !full) : !full
  %e3 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e3 : !full) -> !full
  return %y : !full
}

// The inverse standard deviation may be pow(sqrt(variance + epsilon), -1).

// CHECK-LABEL: func.func @batch_norm_pow_of_standard_deviation
// CHECK:         hip.batch_norm
// CHECK-NOT:     hip.{{sub|mul|pow}}
// CHECK:         return
func.func @batch_norm_pow_of_standard_deviation(%ctx: !hip.context, %x: !full,
                                                %mean: !vec, %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.pow(%ctx) ins(%sd : !vec) outs(%e2 : !vec) {exponent = -1.000000e+00 : f64} : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y : !full
}

// Epsilon is stored as an f32 attribute, rounded from the type the graph added
// it in. Here it is an f16 value.

// CHECK-LABEL: func.func @batch_norm_f16_epsilon
// CHECK:         hip.batch_norm
// CHECK-SAME:    {epsilon = 1.00135803E-5 : f32}
func.func @batch_norm_f16_epsilon(%ctx: !hip.context, %x: tensor<1x4x2x2xf16>,
                                  %mean: tensor<4xf16>, %var: tensor<4xf16>)
    -> tensor<1x4x2x2xf16> {
  %eps_f64 = arith.constant 1.000000e-05 : f64
  %eps_f16 = arith.truncf %eps_f64 : f64 to f16
  %eps = tensor.splat %eps_f16 : tensor<f16>
  %e0 = tensor.empty() : tensor<4xf16>
  %vpe = hip.add(%ctx) ins(%var, %eps : tensor<4xf16>, tensor<f16>) outs(%e0 : tensor<4xf16>) -> tensor<4xf16>
  %e1 = tensor.empty() : tensor<4xf16>
  %sd = hip.sqrt(%ctx) ins(%vpe : tensor<4xf16>) outs(%e1 : tensor<4xf16>) : tensor<4xf16>
  %e2 = tensor.empty() : tensor<4xf16>
  %inv = hip.reciprocal(%ctx) ins(%sd : tensor<4xf16>) outs(%e2 : tensor<4xf16>) : tensor<4xf16>

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf16> into tensor<4x1x1xf16>
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf16> into tensor<4x1x1xf16>

  %e3 = tensor.empty() : tensor<1x4x2x2xf16>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>) outs(%e3 : tensor<1x4x2x2xf16>) : tensor<1x4x2x2xf16>
  %e4 = tensor.empty() : tensor<1x4x2x2xf16>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<1x4x2x2xf16>, tensor<4x1x1xf16>) outs(%e4 : tensor<1x4x2x2xf16>) -> tensor<1x4x2x2xf16>
  return %y : tensor<1x4x2x2xf16>
}

// A truncation of epsilon that rounds upward is reproduced.

// CHECK-LABEL: func.func @batch_norm_epsilon_rounding_mode
// CHECK:         hip.batch_norm
// CHECK-SAME:    {epsilon = 1.00000007E-5 : f32}
func.func @batch_norm_epsilon_rounding_mode(%ctx: !hip.context, %x: !full,
                                            %mean: !vec, %var: !vec) -> !full {
  %eps_f64 = arith.constant 1.000000e-05 : f64
  %eps_f32 = arith.truncf %eps_f64 upward : f64 to f32
  %eps = tensor.splat %eps_f32 : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y : !full
}

// An f64 epsilon that f32 cannot represent exactly is rounded.

// CHECK-LABEL: func.func @batch_norm_f64_epsilon
// CHECK:         hip.batch_norm
// CHECK-SAME:    {epsilon = 1.000000e-03 : f32}
func.func @batch_norm_f64_epsilon(%ctx: !hip.context, %x: tensor<1x4x2x2xf64>,
                                  %mean: tensor<4xf64>, %var: tensor<4xf64>)
    -> tensor<1x4x2x2xf64> {
  %eps_f64 = arith.constant 1.000000e-03 : f64
  %eps = tensor.splat %eps_f64 : tensor<f64>
  %e0 = tensor.empty() : tensor<4xf64>
  %vpe = hip.add(%ctx) ins(%var, %eps : tensor<4xf64>, tensor<f64>) outs(%e0 : tensor<4xf64>) -> tensor<4xf64>
  %e1 = tensor.empty() : tensor<4xf64>
  %sd = hip.sqrt(%ctx) ins(%vpe : tensor<4xf64>) outs(%e1 : tensor<4xf64>) : tensor<4xf64>
  %e2 = tensor.empty() : tensor<4xf64>
  %inv = hip.reciprocal(%ctx) ins(%sd : tensor<4xf64>) outs(%e2 : tensor<4xf64>) : tensor<4xf64>

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf64> into tensor<4x1x1xf64>
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf64> into tensor<4x1x1xf64>

  %e3 = tensor.empty() : tensor<1x4x2x2xf64>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<1x4x2x2xf64>, tensor<4x1x1xf64>) outs(%e3 : tensor<1x4x2x2xf64>) : tensor<1x4x2x2xf64>
  %e4 = tensor.empty() : tensor<1x4x2x2xf64>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<1x4x2x2xf64>, tensor<4x1x1xf64>) outs(%e4 : tensor<1x4x2x2xf64>) -> tensor<1x4x2x2xf64>
  return %y : tensor<1x4x2x2xf64>
}

// The batch dimension is dynamic, and the destination of the last op is sized
// from a tensor.dim of an intermediate. The fusion erases that intermediate, so
// the query is retargeted to x.

// CHECK-LABEL: func.func @batch_norm_dynamic_batch
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x4x2x2xf32>
// CHECK-NOT:     hip.sub
// CHECK:         %[[DIM:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x4x2x2xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty(%[[DIM]]) : tensor<?x4x2x2xf32>
// CHECK:         hip.batch_norm(%[[CTX]]) ins(%[[X]], {{.*}} outs(%[[INIT]] : tensor<?x4x2x2xf32>)
// CHECK-NOT:     hip.{{sub|mul}}
func.func @batch_norm_dynamic_batch(%ctx: !hip.context, %x: tensor<?x4x2x2xf32>,
                                    %mean: !vec, %var: !vec) -> tensor<?x4x2x2xf32> {
  %c0 = arith.constant 0 : index
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %n = tensor.dim %x, %c0 : tensor<?x4x2x2xf32>
  %e3 = tensor.empty(%n) : tensor<?x4x2x2xf32>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<?x4x2x2xf32>, !view) outs(%e3 : tensor<?x4x2x2xf32>) : tensor<?x4x2x2xf32>
  %cn = tensor.dim %c, %c0 : tensor<?x4x2x2xf32>
  %e4 = tensor.empty(%cn) : tensor<?x4x2x2xf32>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<?x4x2x2xf32>, !view) outs(%e4 : tensor<?x4x2x2xf32>) -> tensor<?x4x2x2xf32>
  return %y : tensor<?x4x2x2xf32>
}

// A full-size add after the batch normalization is not part of it and stays.

// CHECK-LABEL: func.func @batch_norm_then_full_size_add
// CHECK:         hip.batch_norm
// CHECK:         hip.add
// CHECK-NOT:     hip.{{sub|mul}}
// CHECK:         return
func.func @batch_norm_then_full_size_add(%ctx: !hip.context, %x: !full, %other: !full,
                                         %mean: !vec, %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %n = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  %e5 = tensor.empty() : !full
  %y = hip.add(%ctx) ins(%n, %other : !full, !full) outs(%e5 : !full) -> !full
  return %y : !full
}

// Cases that must not fuse.

// A view on another dimension than the channel dimension, although it has C
// elements.

// CHECK-LABEL: func.func @no_fusion_view_not_on_channel
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_view_not_on_channel(%ctx: !hip.context, %x: tensor<1x4x4x4xf32>,
                                         %mean: !vec, %var: !vec) -> tensor<1x4x4x4xf32> {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [1, 1, 4] : !vec into tensor<1x1x4xf32>
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [1, 1, 4] : !vec into tensor<1x1x4xf32>

  %e3 = tensor.empty() : tensor<1x4x4x4xf32>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>) outs(%e3 : tensor<1x4x4x4xf32>) : tensor<1x4x4x4xf32>
  %e4 = tensor.empty() : tensor<1x4x4x4xf32>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<1x4x4x4xf32>, tensor<1x1x4xf32>) outs(%e4 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %y : tensor<1x4x4x4xf32>
}

// The centered input is also returned, so the chain must stay to produce it.

// CHECK-LABEL: func.func @no_fusion_intermediate_has_other_use
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_intermediate_has_other_use(%ctx: !hip.context, %x: !full,
                                                %mean: !vec, %var: !vec)
    -> (!full, !full) {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y, %c : !full, !full
}

// mean - x is the negated centering, not the batch normalization's x - mean.

// CHECK-LABEL: func.func @no_fusion_reversed_subtraction
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_reversed_subtraction(%ctx: !hip.context, %x: !full, %mean: !vec,
                                          %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%mean_v, %x : !view, !full) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y : !full
}

// Epsilon is a run-time value, which hip.batch_norm cannot take as an attribute.

// CHECK-LABEL: func.func @no_fusion_runtime_epsilon
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_runtime_epsilon(%ctx: !hip.context, %x: !full, %mean: !vec,
                                     %var: !vec, %eps: tensor<f32>) -> !full {
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %sd = hip.sqrt(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) : !vec
  %e2 = tensor.empty() : !vec
  %inv = hip.reciprocal(%ctx) ins(%sd : !vec) outs(%e2 : !vec) : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e3 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e3 : !full) : !full
  %e4 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e4 : !full) -> !full
  return %y : !full
}

// An f64 epsilon of 1e-50 underflows the f32 attribute to zero, which turns the
// result for a zero variance from finite into infinity.

// CHECK-LABEL: func.func @no_fusion_epsilon_underflows_f32
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_epsilon_underflows_f32(%ctx: !hip.context,
                                            %x: tensor<1x4x2x2xf64>,
                                            %mean: tensor<4xf64>,
                                            %var: tensor<4xf64>)
    -> tensor<1x4x2x2xf64> {
  %eps_f64 = arith.constant 1.000000e-50 : f64
  %eps = tensor.splat %eps_f64 : tensor<f64>
  %e0 = tensor.empty() : tensor<4xf64>
  %vpe = hip.add(%ctx) ins(%var, %eps : tensor<4xf64>, tensor<f64>) outs(%e0 : tensor<4xf64>) -> tensor<4xf64>
  %e1 = tensor.empty() : tensor<4xf64>
  %sd = hip.sqrt(%ctx) ins(%vpe : tensor<4xf64>) outs(%e1 : tensor<4xf64>) : tensor<4xf64>
  %e2 = tensor.empty() : tensor<4xf64>
  %inv = hip.reciprocal(%ctx) ins(%sd : tensor<4xf64>) outs(%e2 : tensor<4xf64>) : tensor<4xf64>

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf64> into tensor<4x1x1xf64>
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : tensor<4xf64> into tensor<4x1x1xf64>

  %e3 = tensor.empty() : tensor<1x4x2x2xf64>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<1x4x2x2xf64>, tensor<4x1x1xf64>) outs(%e3 : tensor<1x4x2x2xf64>) : tensor<1x4x2x2xf64>
  %e4 = tensor.empty() : tensor<1x4x2x2xf64>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<1x4x2x2xf64>, tensor<4x1x1xf64>) outs(%e4 : tensor<1x4x2x2xf64>) -> tensor<1x4x2x2xf64>
  return %y : tensor<1x4x2x2xf64>
}

// An exponent of -0.3 is not an inverse standard deviation.

// CHECK-LABEL: func.func @no_fusion_pow_exponent
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.pow
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_pow_exponent(%ctx: !hip.context, %x: !full, %mean: !vec,
                                  %var: !vec) -> !full {
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty() : !vec
  %vpe = hip.add(%ctx) ins(%var, %eps : !vec, tensor<f32>) outs(%e0 : !vec) -> !vec
  %e1 = tensor.empty() : !vec
  %inv = hip.pow(%ctx) ins(%vpe : !vec) outs(%e1 : !vec) {exponent = -3.000000e-01 : f64} : !vec

  %mean_v = tensor.expand_shape %mean [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view
  %inv_v = tensor.expand_shape %inv [[0, 1, 2]] output_shape [4, 1, 1] : !vec into !view

  %e2 = tensor.empty() : !full
  %c = hip.sub(%ctx) ins(%x, %mean_v : !full, !view) outs(%e2 : !full) : !full
  %e3 = tensor.empty() : !full
  %y = hip.mul(%ctx) ins(%c, %inv_v : !full, !view) outs(%e3 : !full) -> !full
  return %y : !full
}

// The channel dimension is dynamic, so the vectors cannot be shown to have as
// many elements as the input has channels.

// CHECK-LABEL: func.func @no_fusion_dynamic_channel
// CHECK-NOT:     hip.batch_norm
// CHECK:         hip.sub
// CHECK-NOT:     hip.batch_norm
// CHECK:         return
func.func @no_fusion_dynamic_channel(%ctx: !hip.context, %x: tensor<1x?x2x2xf32>,
                                     %mean_v: tensor<?x1x1xf32>,
                                     %var_v: tensor<?x1x1xf32>) -> tensor<1x?x2x2xf32> {
  %c1 = arith.constant 1 : index
  %channels = tensor.dim %x, %c1 : tensor<1x?x2x2xf32>
  %eps = hip.constant {value = dense<1.0e-5> : tensor<f32>} : tensor<f32>
  %e0 = tensor.empty(%channels) : tensor<?x1x1xf32>
  %vpe = hip.add(%ctx) ins(%var_v, %eps : tensor<?x1x1xf32>, tensor<f32>) outs(%e0 : tensor<?x1x1xf32>) -> tensor<?x1x1xf32>
  %e1 = tensor.empty(%channels) : tensor<?x1x1xf32>
  %sd = hip.sqrt(%ctx) ins(%vpe : tensor<?x1x1xf32>) outs(%e1 : tensor<?x1x1xf32>) : tensor<?x1x1xf32>
  %e2 = tensor.empty(%channels) : tensor<?x1x1xf32>
  %inv_v = hip.reciprocal(%ctx) ins(%sd : tensor<?x1x1xf32>) outs(%e2 : tensor<?x1x1xf32>) : tensor<?x1x1xf32>

  %e3 = tensor.empty(%channels) : tensor<1x?x2x2xf32>
  %c = hip.sub(%ctx) ins(%x, %mean_v : tensor<1x?x2x2xf32>, tensor<?x1x1xf32>) outs(%e3 : tensor<1x?x2x2xf32>) : tensor<1x?x2x2xf32>
  %e4 = tensor.empty(%channels) : tensor<1x?x2x2xf32>
  %y = hip.mul(%ctx) ins(%c, %inv_v : tensor<1x?x2x2xf32>, tensor<?x1x1xf32>) outs(%e4 : tensor<1x?x2x2xf32>) -> tensor<1x?x2x2xf32>
  return %y : tensor<1x?x2x2xf32>
}
