// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Sigmoid changes values, so the input and output quantization parameters are
// deliberately different throughout: requantization is part of the fused op,
// and the checks have to prove both pairs reach the attributes.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @qsigmoid
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x2048xui16>) -> tensor<1x128x2048xui16> {
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x128x2048xui16>
// CHECK-NEXT:    %[[QSIG:.*]] = hip.qsigmoid(%[[CTX]]) ins(%[[X]] : tensor<1x128x2048xui16>) outs(%[[INIT]] : tensor<1x128x2048xui16>) {input_scale = 1.000000e-01 : f32, input_zp = 35189 : i64, output_scale = 2.000000e-01 : f32, output_zp = 35274 : i64} : tensor<1x128x2048xui16>
// CHECK-NEXT:    return %[[QSIG]] : tensor<1x128x2048xui16>
// CHECK-NEXT:  }
func.func @qsigmoid(%ctx: !hip.context,
                    %x: tensor<1x128x2048xui16>) -> tensor<1x128x2048xui16> {
  %in_s = hip.constant {value = dense<1.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.000000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x128x2048xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x128x2048xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<1x128x2048xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x2048xf32>

  %e1 = tensor.empty() : tensor<1x128x2048xf32>
  %s = hip.sigmoid(%ctx) ins(%dq : tensor<1x128x2048xf32>)
      outs(%e1 : tensor<1x128x2048xf32>) : tensor<1x128x2048xf32>

  %e2 = tensor.empty() : tensor<1x128x2048xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%s, %out_s : tensor<1x128x2048xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<1x128x2048xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x2048xui16>

  return %q : tensor<1x128x2048xui16>
}

// -----

// A dynamic batch fuses too. The replacement's init takes its dynamic extent
// from hip.sigmoid's init rather than from the result type, and the fused op
// reads its element count from the memref descriptor once lowered, so nothing
// here needs a static shape. The checks stay loose about how that extent is
// recovered because only the fusion itself is the subject.
// CHECK-LABEL: func.func @qsigmoid_dynamic_batch
// CHECK-NOT:     hip.sigmoid
// CHECK:         %[[INIT:.*]] = tensor.empty(%{{.*}}) : tensor<?x128x2048xui16>
// CHECK:         %[[QSIG:.*]] = hip.qsigmoid(%{{.*}}) ins(%{{.*}} : tensor<?x128x2048xui16>) outs(%[[INIT]] : tensor<?x128x2048xui16>) {input_scale = 1.000000e-01 : f32, input_zp = 35189 : i64, output_scale = 2.000000e-01 : f32, output_zp = 35274 : i64} : tensor<?x128x2048xui16>
// CHECK:         return %[[QSIG]] : tensor<?x128x2048xui16>
func.func @qsigmoid_dynamic_batch(%ctx: !hip.context,
                                  %x: tensor<?x128x2048xui16>) -> tensor<?x128x2048xui16> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x128x2048xui16>
  %in_s = hip.constant {value = dense<1.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<35189> : tensor<ui16>} : tensor<ui16>
  %out_s = hip.constant {value = dense<2.000000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<35274> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty(%n) : tensor<?x128x2048xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<?x128x2048xui16>, tensor<f32>)
      zero_point(%in_z : tensor<ui16>)
      outs(%e0 : tensor<?x128x2048xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x128x2048xf32>

  %e1 = tensor.empty(%n) : tensor<?x128x2048xf32>
  %s = hip.sigmoid(%ctx) ins(%dq : tensor<?x128x2048xf32>)
      outs(%e1 : tensor<?x128x2048xf32>) : tensor<?x128x2048xf32>

  %e2 = tensor.empty(%n) : tensor<?x128x2048xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%s, %out_s : tensor<?x128x2048xf32>, tensor<f32>)
      zero_point(%out_z : tensor<ui16>)
      outs(%e2 : tensor<?x128x2048xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<?x128x2048xui16>

  return %q : tensor<?x128x2048xui16>
}

// -----

// An 8-bit sandwich is not the supported width, and there is no 8-bit
// qsigmoid pattern for it to fall through to.
// CHECK-LABEL: func.func @qsigmoid_i8_not_fused
// CHECK-NOT:   hip.qsigmoid
func.func @qsigmoid_i8_not_fused(%ctx: !hip.context,
                                 %x: tensor<1x8x16xi8>) -> tensor<1x8x16xi8> {
  %in_s = hip.constant {value = dense<1.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %out_s = hip.constant {value = dense<2.000000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<1> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<1x8x16xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x8x16xi8>, tensor<f32>)
      zero_point(%in_z : tensor<i8>)
      outs(%e0 : tensor<1x8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x16xf32>

  %e1 = tensor.empty() : tensor<1x8x16xf32>
  %s = hip.sigmoid(%ctx) ins(%dq : tensor<1x8x16xf32>)
      outs(%e1 : tensor<1x8x16xf32>) : tensor<1x8x16xf32>

  %e2 = tensor.empty() : tensor<1x8x16xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%s, %out_s : tensor<1x8x16xf32>, tensor<f32>)
      zero_point(%out_z : tensor<i8>)
      outs(%e2 : tensor<1x8x16xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<1x8x16xi8>

  return %q : tensor<1x8x16xi8>
}

// -----

// A signed 16-bit sandwich has the supported width but not the supported
// signedness: it spans a different range and widens differently, so it stays
// unfused rather than reaching a wrap that reads the codes as unsigned.
// CHECK-LABEL: func.func @qsigmoid_i16_not_fused
// CHECK-NOT:   hip.qsigmoid
func.func @qsigmoid_i16_not_fused(%ctx: !hip.context,
                                  %x: tensor<1x8x16xi16>) -> tensor<1x8x16xi16> {
  %in_s = hip.constant {value = dense<1.000000e-01> : tensor<f32>} : tensor<f32>
  %in_z = hip.constant {value = dense<-5> : tensor<i16>} : tensor<i16>
  %out_s = hip.constant {value = dense<2.000000e-01> : tensor<f32>} : tensor<f32>
  %out_z = hip.constant {value = dense<7> : tensor<i16>} : tensor<i16>

  %e0 = tensor.empty() : tensor<1x8x16xf32>
  %dq = hip.dequantize_linear(%ctx)
      ins(%x, %in_s : tensor<1x8x16xi16>, tensor<f32>)
      zero_point(%in_z : tensor<i16>)
      outs(%e0 : tensor<1x8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x8x16xf32>

  %e1 = tensor.empty() : tensor<1x8x16xf32>
  %s = hip.sigmoid(%ctx) ins(%dq : tensor<1x8x16xf32>)
      outs(%e1 : tensor<1x8x16xf32>) : tensor<1x8x16xf32>

  %e2 = tensor.empty() : tensor<1x8x16xi16>
  %q = hip.quantize_linear(%ctx)
      ins(%s, %out_s : tensor<1x8x16xf32>, tensor<f32>)
      zero_point(%out_z : tensor<i16>)
      outs(%e2 : tensor<1x8x16xi16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x8x16xi16>

  return %q : tensor<1x8x16xi16>
}
