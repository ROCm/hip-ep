// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// HIP normalization ops expand to TOSA reduce_sum / rsqrt / mul / add inside
// a rock.kernel function. That covers the seven ONNX/contrib names:
// LayerNormalization -> hip.layer_norm
// InstanceNormalization -> hip.instance_norm
// SimplifiedLayerNormalization / RMSNormalization -> hip.rms_norm
// SkipSimplifiedLayerNormalization -> hip.skip_rms_norm
// SkipLayerNormalization -> hip.add + hip.layer_norm (both convert)
// LpNormalization is not a HIP op: L2 on the last static axis fuses to
// hip.rms_norm; other cases decompose to mul/reduce/sqrt/div before HIP.
// Quantized QDQ LpNormalization (hip.qlpnormalization) is not converted.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @rms_norm
// CHECK: tosa.mul
// CHECK: tosa.reduce_sum
// CHECK: tosa.rsqrt
// CHECK: tosa.mul
// CHECK-NOT: hip.rms_norm
func.func @rms_norm(%ctx: !hip.context, %x: tensor<2x4xf16>,
                    %scale: tensor<4xf16>, %init: tensor<2x4xf16>)
    -> tensor<2x4xf16> attributes {rock.kernel} {
  %r = hip.rms_norm(%ctx)
      ins(%x, %scale : tensor<2x4xf16>, tensor<4xf16>)
      outs(%init : tensor<2x4xf16>)
      {axis = -1 : i64, epsilon = 1.000000e-05 : f32, stash_type = 1 : i64}
      : tensor<2x4xf16>
  return %r : tensor<2x4xf16>
}

// CHECK-LABEL: func.func @layer_norm
// CHECK: tosa.reduce_sum
// CHECK: tosa.sub
// CHECK: tosa.rsqrt
// CHECK: tosa.add
// CHECK-NOT: hip.layer_norm
func.func @layer_norm(%ctx: !hip.context, %x: tensor<2x3x4xf32>,
                      %scale: tensor<4xf32>, %bias: tensor<4xf32>,
                      %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    attributes {rock.kernel} {
  %r = hip.layer_norm(%ctx)
      ins(%x, %scale, %bias : tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%init : tensor<2x3x4xf32>)
      {axis = -1 : i64, epsilon = 1.000000e-05 : f32, stash_type = 1 : i64}
      : tensor<2x3x4xf32>
  return %r : tensor<2x3x4xf32>
}

// SkipLayerNormalization is hip.add + hip.layer_norm.
// CHECK-LABEL: func.func @skip_layer_norm
// CHECK: tosa.add
// CHECK: tosa.sub
// CHECK: tosa.rsqrt
// CHECK-NOT: hip.add
// CHECK-NOT: hip.layer_norm
func.func @skip_layer_norm(%ctx: !hip.context, %x: tensor<1x4x8xf16>,
                           %skip: tensor<1x4x8xf16>, %gamma: tensor<8xf16>,
                           %beta: tensor<8xf16>, %sum_init: tensor<1x4x8xf16>,
                           %init: tensor<1x4x8xf16>) -> tensor<1x4x8xf16>
    attributes {rock.kernel} {
  %sum = hip.add(%ctx) ins(%x, %skip : tensor<1x4x8xf16>, tensor<1x4x8xf16>)
                       outs(%sum_init : tensor<1x4x8xf16>) -> tensor<1x4x8xf16>
  %r = hip.layer_norm(%ctx)
      ins(%sum, %gamma, %beta : tensor<1x4x8xf16>, tensor<8xf16>, tensor<8xf16>)
      outs(%init : tensor<1x4x8xf16>)
      {axis = -1 : i64, epsilon = 1.000000e-05 : f32, stash_type = 1 : i64}
      : tensor<1x4x8xf16>
  return %r : tensor<1x4x8xf16>
}

// CHECK-LABEL: func.func @instance_norm
// CHECK: tosa.reduce_sum
// CHECK: tosa.sub
// CHECK: tosa.rsqrt
// CHECK-NOT: hip.instance_norm
func.func @instance_norm(%ctx: !hip.context, %x: tensor<2x3x4x4xf32>,
                         %scale: tensor<3xf32>, %bias: tensor<3xf32>,
                         %init: tensor<2x3x4x4xf32>) -> tensor<2x3x4x4xf32>
    attributes {rock.kernel} {
  %r = hip.instance_norm(%ctx)
      ins(%x, %scale, %bias : tensor<2x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>)
      outs(%init : tensor<2x3x4x4xf32>)
      {epsilon = 1.000000e-05 : f32}
      : tensor<2x3x4x4xf32>
  return %r : tensor<2x3x4x4xf32>
}

// CHECK-LABEL: func.func @skip_rms_norm
// CHECK: tosa.add
// CHECK: tosa.rsqrt
// CHECK-NOT: hip.skip_rms_norm
func.func @skip_rms_norm(%ctx: !hip.context, %x: tensor<2x4xf16>,
                         %skip: tensor<2x4xf16>, %gamma: tensor<4xf16>,
                         %init: tensor<2x4xf16>) -> tensor<2x4xf16>
    attributes {rock.kernel} {
  %r = hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %gamma : tensor<2x4xf16>, tensor<2x4xf16>, tensor<4xf16>)
      outs(%init : tensor<2x4xf16>)
      {epsilon = 1.000000e-05 : f32}
      : tensor<2x4xf16>
  return %r : tensor<2x4xf16>
}

// CHECK-LABEL: func.func @rms_norm_suffix_scale
// CHECK: tosa.reshape
// CHECK: tosa.rsqrt
// CHECK-NOT: hip.rms_norm
func.func @rms_norm_suffix_scale(%ctx: !hip.context, %x: tensor<2x3x4xf32>,
                                 %scale: tensor<3x4xf32>,
                                 %init: tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
    attributes {rock.kernel} {
  %r = hip.rms_norm(%ctx)
      ins(%x, %scale : tensor<2x3x4xf32>, tensor<3x4xf32>)
      outs(%init : tensor<2x3x4xf32>)
      {axis = -2 : i64, epsilon = 1.000000e-05 : f32, stash_type = 0 : i64}
      : tensor<2x3x4xf32>
  return %r : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @rms_norm_flat_suffix_scale
// CHECK: tosa.reshape
// CHECK: tosa.rsqrt
// CHECK-NOT: hip.rms_norm
func.func @rms_norm_flat_suffix_scale(%ctx: !hip.context, %x: tensor<2x3x4xf32>,
                                      %scale: tensor<12xf32>,
                                      %init: tensor<2x3x4xf32>)
    -> tensor<2x3x4xf32> attributes {rock.kernel} {
  %r = hip.rms_norm(%ctx)
      ins(%x, %scale : tensor<2x3x4xf32>, tensor<12xf32>)
      outs(%init : tensor<2x3x4xf32>)
      {axis = -2 : i64, epsilon = 1.000000e-05 : f32, stash_type = 0 : i64}
      : tensor<2x3x4xf32>
  return %r : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @layer_norm_suffix_scale
// CHECK: tosa.reshape
// CHECK: tosa.add
// CHECK-NOT: hip.layer_norm
func.func @layer_norm_suffix_scale(%ctx: !hip.context, %x: tensor<2x3x4xf32>,
                                   %scale: tensor<3x4xf32>,
                                   %bias: tensor<3x4xf32>,
                                   %init: tensor<2x3x4xf32>)
    -> tensor<2x3x4xf32> attributes {rock.kernel} {
  %r = hip.layer_norm(%ctx)
      ins(%x, %scale, %bias : tensor<2x3x4xf32>, tensor<3x4xf32>,
                              tensor<3x4xf32>)
      outs(%init : tensor<2x3x4xf32>)
      {axis = -2 : i64, epsilon = 1.000000e-05 : f32, stash_type = 1 : i64}
      : tensor<2x3x4xf32>
  return %r : tensor<2x3x4xf32>
}
