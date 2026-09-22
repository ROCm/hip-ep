// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The middle op is hip.rms_norm rather than an LpNormalization: nothing of
// that name reaches this layer, because LpNormalizationConversion decomposes
// onnx.LpNormalization during pre-lowering and its fast path emits a
// SimplifiedLayerNormalization for exactly the p=2 trailing-axis case this
// pattern fuses.
//
// What licenses the fusion is the identity rms_norm(x, 1/sqrt(N), 0) = L2(x),
// so the trailing extent is fixed at 64 throughout and the rms scale is its
// 1/sqrt(N) = 0.125. The two negative cases at the end perturb one half of
// that identity each, which is what separates a real RMS norm from an L2 one.
//
// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @qlpnormalization
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x64xui16>) -> tensor<1x128x64xui16> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x128x64xui16>
// CHECK-NEXT:    %[[QLP:.*]] = hip.qlpnormalization(%[[CTX]]) ins(%[[X]] : tensor<1x128x64xui16>) outs(%[[INIT]] : tensor<1x128x64xui16>) {axis = -1 : i64, input_scale = 5.000000e-01 : f32, input_zp = 35189 : i64, output_scale = 2.500000e-01 : f32, output_zp = 35274 : i64, p = 2 : i64} : tensor<1x128x64xui16>
// CHECK-NEXT:    return %[[QLP]] : tensor<1x128x64xui16>
// CHECK-NEXT:  }
func.func @qlpnormalization(%ctx: !hip.context,
                            %x: tensor<1x128x64xui16>) -> tensor<1x128x64xui16> {
  %in_s = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>
  %rms_s = hip.constant {value = dense<1.250000e-01> : tensor<64xf32>} : tensor<64xf32>

  %e0 = tensor.empty() : tensor<1x128x64xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x128x64xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<1x128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x64xf32>

  %e1 = tensor.empty() : tensor<1x128x64xf32>
  %rms = hip.rms_norm(%ctx)
      ins(%dq, %rms_s : tensor<1x128x64xf32>, tensor<64xf32>)
      outs(%e1 : tensor<1x128x64xf32>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64}
      : tensor<1x128x64xf32>

  %e2 = tensor.empty() : tensor<1x128x64xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%rms, %out_s : tensor<1x128x64xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<1x128x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x64xui16>

  return %q : tensor<1x128x64xui16>
}

// -----

// Only the reduced extent has to be static, because that is all the 1/sqrt(N)
// comparison needs; BuildInit recovers the batch dimension. The ONNX-layer
// pattern required every dimension static, so this graph is one the migration
// newly fuses.
// CHECK-LABEL: func.func @qlpnormalization_dynamic_batch
// CHECK:         %[[INIT:.*]] = tensor.empty(%{{.*}}) : tensor<?x128x64xui16>
// CHECK:         %[[QLP:.*]] = hip.qlpnormalization(%{{.*}}) ins(%{{.*}} : tensor<?x128x64xui16>) outs(%[[INIT]] : tensor<?x128x64xui16>) {axis = -1 : i64, input_scale = 5.000000e-01 : f32, input_zp = 35189 : i64, output_scale = 2.500000e-01 : f32, output_zp = 35274 : i64, p = 2 : i64} : tensor<?x128x64xui16>
// CHECK:         return %[[QLP]] : tensor<?x128x64xui16>
func.func @qlpnormalization_dynamic_batch(%ctx: !hip.context,
                                          %x: tensor<?x128x64xui16>) -> tensor<?x128x64xui16> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x128x64xui16>
  %in_s = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>
  %rms_s = hip.constant {value = dense<1.250000e-01> : tensor<64xf32>} : tensor<64xf32>

  %e0 = tensor.empty(%n) : tensor<?x128x64xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<?x128x64xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<?x128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x128x64xf32>

  %e1 = tensor.empty(%n) : tensor<?x128x64xf32>
  %rms = hip.rms_norm(%ctx)
      ins(%dq, %rms_s : tensor<?x128x64xf32>, tensor<64xf32>)
      outs(%e1 : tensor<?x128x64xf32>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64}
      : tensor<?x128x64xf32>

  %e2 = tensor.empty(%n) : tensor<?x128x64xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%rms, %out_s : tensor<?x128x64xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<?x128x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<?x128x64xui16>

  return %q : tensor<?x128x64xui16>
}

// -----

// A nonzero epsilon breaks the identity: the denominator is no longer the
// plain L2 norm. This is a genuine RMS norm that happens to sit in a Q/DQ
// sandwich, and hip.qlpnormalization has no epsilon to carry it over to.
// CHECK-LABEL: func.func @rms_norm_nonzero_epsilon_not_fused
// CHECK-NOT:   hip.qlpnormalization
func.func @rms_norm_nonzero_epsilon_not_fused(%ctx: !hip.context,
                                              %x: tensor<1x128x64xui16>) -> tensor<1x128x64xui16> {
  %in_s = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>
  %rms_s = hip.constant {value = dense<1.250000e-01> : tensor<64xf32>} : tensor<64xf32>

  %e0 = tensor.empty() : tensor<1x128x64xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x128x64xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<1x128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x64xf32>

  %e1 = tensor.empty() : tensor<1x128x64xf32>
  %rms = hip.rms_norm(%ctx)
      ins(%dq, %rms_s : tensor<1x128x64xf32>, tensor<64xf32>)
      outs(%e1 : tensor<1x128x64xf32>)
      {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
      : tensor<1x128x64xf32>

  %e2 = tensor.empty() : tensor<1x128x64xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%rms, %out_s : tensor<1x128x64xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<1x128x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x64xui16>

  return %q : tensor<1x128x64xui16>
}

// -----

// Unit scale instead of 1/sqrt(64): still splat, still epsilon 0, but the
// result is rescaled by sqrt(N) against an L2 norm. Perturbing the value
// rather than the splatness is what pins the check to the identity instead of
// to the shape of the constant.
// CHECK-LABEL: func.func @rms_norm_unit_scale_not_fused
// CHECK-NOT:   hip.qlpnormalization
func.func @rms_norm_unit_scale_not_fused(%ctx: !hip.context,
                                         %x: tensor<1x128x64xui16>) -> tensor<1x128x64xui16> {
  %in_s = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>
  %rms_s = hip.constant {value = dense<1.000000e+00> : tensor<64xf32>} : tensor<64xf32>

  %e0 = tensor.empty() : tensor<1x128x64xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x128x64xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<1x128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x64xf32>

  %e1 = tensor.empty() : tensor<1x128x64xf32>
  %rms = hip.rms_norm(%ctx)
      ins(%dq, %rms_s : tensor<1x128x64xf32>, tensor<64xf32>)
      outs(%e1 : tensor<1x128x64xf32>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64}
      : tensor<1x128x64xf32>

  %e2 = tensor.empty() : tensor<1x128x64xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%rms, %out_s : tensor<1x128x64xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<1x128x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x64xui16>

  return %q : tensor<1x128x64xui16>
}

// -----

// hip.qlpnormalization is a UINT16 kernel, so signed 16-bit storage has no
// fused form to fall through to.
// CHECK-LABEL: func.func @qlpnormalization_i16_not_fused
// CHECK-NOT:   hip.qlpnormalization
func.func @qlpnormalization_i16_not_fused(%ctx: !hip.context,
                                          %x: tensor<1x128x64xi16>) -> tensor<1x128x64xi16> {
  %in_s = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<-1200> : tensor<i16>} : tensor<i16>
  %out_s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<-2300> : tensor<i16>} : tensor<i16>
  %rms_s = hip.constant {value = dense<1.250000e-01> : tensor<64xf32>} : tensor<64xf32>

  %e0 = tensor.empty() : tensor<1x128x64xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x128x64xi16>, tensor<f32>)
      zero_point(%in_z : tensor<i16>)
      outs(%e0 : tensor<1x128x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x64xf32>

  %e1 = tensor.empty() : tensor<1x128x64xf32>
  %rms = hip.rms_norm(%ctx)
      ins(%dq, %rms_s : tensor<1x128x64xf32>, tensor<64xf32>)
      outs(%e1 : tensor<1x128x64xf32>)
      {axis = -1 : i64, epsilon = 0.000000e+00 : f32, stash_type = 1 : i64}
      : tensor<1x128x64xf32>

  %e2 = tensor.empty() : tensor<1x128x64xi16>
  %q = hip.quantize_linear(%ctx)
      ins(%rms, %out_s : tensor<1x128x64xf32>, tensor<f32>)
      zero_point(%out_z : tensor<i16>)
      outs(%e2 : tensor<1x128x64xi16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x64xi16>

  return %q : tensor<1x128x64xi16>
}
