// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Q/DQ around hip.add fold into the add's element types: the same hip.add as in
// a float model, quantized by type.

// CHECK-LABEL: func.func @qadd
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[LHS:.*]]: tensor<1x128x32xi8>, %[[RHS:.*]]: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
// CHECK-NEXT:    %[[L:.*]] = hip.storage_cast %[[LHS]] : tensor<1x128x32xi8> to tensor<1x128x32x!quant.uniform<i8:f32, 2.500000e-01:-5>>
// CHECK-NEXT:    %[[R:.*]] = hip.storage_cast %[[RHS]] : tensor<1x128x32xi8> to tensor<1x128x32x!quant.uniform<i8:f32, 5.000000e-01:3>>
// CHECK-NEXT:    %[[INIT:.*]] = tensor.empty() : tensor<1x128x32x!quant.uniform<i8:f32, 1.250000e-01:7>>
// CHECK-NEXT:    %[[ADD:.*]] = hip.add(%[[CTX]]) ins(%[[L]], %[[R]] : tensor<1x128x32x!quant.uniform<i8:f32, 2.500000e-01:-5>>, tensor<1x128x32x!quant.uniform<i8:f32, 5.000000e-01:3>>) outs(%[[INIT]] : tensor<1x128x32x!quant.uniform<i8:f32, 1.250000e-01:7>>) -> tensor<1x128x32x!quant.uniform<i8:f32, 1.250000e-01:7>>
// CHECK-NEXT:    %[[OUT:.*]] = hip.storage_cast %[[ADD]] : tensor<1x128x32x!quant.uniform<i8:f32, 1.250000e-01:7>> to tensor<1x128x32xi8>
// CHECK-NEXT:    return %[[OUT]] : tensor<1x128x32xi8>
// CHECK-NEXT:  }
func.func @qadd(%ctx: !hip.context,
                %lhs: tensor<1x128x32xi8>,
                %rhs: tensor<1x128x32xi8>) -> tensor<1x128x32xi8> {
  %lhs_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %lhs_zp = hip.constant {value = dense<-5> : tensor<i8>} : tensor<i8>
  %rhs_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %rhs_zp = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %out_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %out_zp = hip.constant {value = dense<7> : tensor<i8>} : tensor<i8>

  %e0 = tensor.empty() : tensor<1x128x32xf32>
  %dq_lhs = hip.dequantize_linear(%ctx)
      ins(%lhs, %lhs_scale : tensor<1x128x32xi8>, tensor<f32>)
      zero_point(%lhs_zp : tensor<i8>)
      outs(%e0 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e1 = tensor.empty() : tensor<1x128x32xf32>
  %dq_rhs = hip.dequantize_linear(%ctx)
      ins(%rhs, %rhs_scale : tensor<1x128x32xi8>, tensor<f32>)
      zero_point(%rhs_zp : tensor<i8>)
      outs(%e1 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e2 = tensor.empty() : tensor<1x128x32xf32>
  %sum = hip.add(%ctx)
      ins(%dq_lhs, %dq_rhs : tensor<1x128x32xf32>, tensor<1x128x32xf32>)
      outs(%e2 : tensor<1x128x32xf32>) -> tensor<1x128x32xf32>

  %e3 = tensor.empty() : tensor<1x128x32xi8>
  %q = hip.quantize_linear(%ctx)
      ins(%sum, %out_scale : tensor<1x128x32xf32>, tensor<f32>)
      zero_point(%out_zp : tensor<i8>)
      outs(%e3 : tensor<1x128x32xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 8 : i64,
       saturate = 1 : i64} : tensor<1x128x32xi8>

  return %q : tensor<1x128x32xi8>
}

// -----

// The quantized add kernel reads both operands at the result's element type,
// so a uint16 activation plus an int8 constant has no kernel and stays an
// unfused dequantize, add, quantize chain.

// CHECK-LABEL: func.func @qadd_mixed_types
// CHECK-NOT:     hip.storage_cast
// CHECK:         hip.add
// CHECK:         hip.quantize_linear
func.func @qadd_mixed_types(%ctx: !hip.context,
                            %lhs: tensor<1x128x32xui16>,
                            %rhs: tensor<1x128x32xi8>) -> tensor<1x128x32xui16> {
  %lhs_scale = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %lhs_zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %rhs_scale = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %rhs_zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %out_scale = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %out_zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<1x128x32xf32>
  %dq_lhs = hip.dequantize_linear(%ctx)
      ins(%lhs, %lhs_scale : tensor<1x128x32xui16>, tensor<f32>)
      zero_point(%lhs_zp : tensor<ui16>)
      outs(%e0 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e1 = tensor.empty() : tensor<1x128x32xf32>
  %dq_rhs = hip.dequantize_linear(%ctx)
      ins(%rhs, %rhs_scale : tensor<1x128x32xi8>, tensor<f32>)
      zero_point(%rhs_zp : tensor<i8>)
      outs(%e1 : tensor<1x128x32xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x128x32xf32>

  %e2 = tensor.empty() : tensor<1x128x32xf32>
  %sum = hip.add(%ctx)
      ins(%dq_lhs, %dq_rhs : tensor<1x128x32xf32>, tensor<1x128x32xf32>)
      outs(%e2 : tensor<1x128x32xf32>) -> tensor<1x128x32xf32>

  %e3 = tensor.empty() : tensor<1x128x32xui16>
  %q = hip.quantize_linear(%ctx)
      ins(%sum, %out_scale : tensor<1x128x32xf32>, tensor<f32>)
      zero_point(%out_zp : tensor<ui16>)
      outs(%e3 : tensor<1x128x32xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 16 : i64,
       saturate = 1 : i64} : tensor<1x128x32xui16>

  return %q : tensor<1x128x32xui16>
}

// -----

// A float add is the same op with float types; nothing to fold.

// CHECK-LABEL: func.func @add_float
// CHECK-NEXT:    tensor.empty() : tensor<8xf32>
// CHECK-NEXT:    hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>) outs(%{{.*}} : tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    return
func.func @add_float(%ctx: !hip.context, %lhs: tensor<8xf32>,
                     %rhs: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %sum = hip.add(%ctx) ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%e : tensor<8xf32>) -> tensor<8xf32>
  return %sum : tensor<8xf32>
}

// -----

// Two quantized adds in a row: the storage casts between them fold, so the
// second add reads the first one's quantized result directly.

// CHECK-LABEL: func.func @qadd_chain
// CHECK:         %[[A0:.*]] = hip.add
// CHECK-SAME:      -> tensor<8x!quant.uniform<u16:f32, 1.250000e-01:32768>>
// CHECK:         %[[A1:.*]] = hip.add(%{{.*}}) ins(%[[A0]], %{{.*}} : tensor<8x!quant.uniform<u16:f32, 1.250000e-01:32768>>, tensor<8x!quant.uniform<u16:f32, 2.500000e-01:32768>>)
// CHECK:         hip.storage_cast %[[A1]] : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>> to tensor<8xui16>
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
func.func @qadd_chain(%ctx: !hip.context, %a: tensor<8xui16>,
                      %b: tensor<8xui16>) -> tensor<8xui16> {
  %s_in = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %s_mid = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %s_out = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s_in : tensor<8xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>) outs(%e0 : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx) ins(%b, %s_in : tensor<8xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>) outs(%e1 : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e2 = tensor.empty() : tensor<8xf32>
  %s0 = hip.add(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>)
      outs(%e2 : tensor<8xf32>) -> tensor<8xf32>
  %e3 = tensor.empty() : tensor<8xui16>
  %q0 = hip.quantize_linear(%ctx) ins(%s0, %s_mid : tensor<8xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>) outs(%e3 : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>

  %e4 = tensor.empty() : tensor<8xf32>
  %dq0 = hip.dequantize_linear(%ctx) ins(%q0, %s_mid : tensor<8xui16>, tensor<f32>)
      zero_point(%zp : tensor<ui16>) outs(%e4 : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e5 = tensor.empty() : tensor<8xf32>
  %s1 = hip.add(%ctx) ins(%dq0, %db : tensor<8xf32>, tensor<8xf32>)
      outs(%e5 : tensor<8xf32>) -> tensor<8xf32>
  %e6 = tensor.empty() : tensor<8xui16>
  %q1 = hip.quantize_linear(%ctx) ins(%s1, %s_out : tensor<8xf32>, tensor<f32>)
      zero_point(%zp : tensor<ui16>) outs(%e6 : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %q1 : tensor<8xui16>
}
