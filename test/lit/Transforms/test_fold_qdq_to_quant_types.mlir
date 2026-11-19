// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-fold-qdq-to-quant-types: DQ(inputs) -> op -> Q around any quant-capable op
// becomes the op on !quant.uniform tensors, retyped in place. One pass for every
// op implementing HipQuantCapableOpInterface; no per-op pattern.

// RUN: hip-mlir-opt --split-input-file --hip-fold-qdq-to-quant-types %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context, %[[A:.*]]: tensor<8xui16>, %[[B:.*]]: tensor<8xui16>) -> tensor<8xui16>
// CHECK:         %[[QA:.*]] = hip.storage_cast %[[A]] : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 2.500000e-01:32768>>
// CHECK:         %[[QB:.*]] = hip.storage_cast %[[B]] : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 1.250000e-01:32768>>
// CHECK:         %[[S:.*]] = hip.add(%[[CTX]]) ins(%[[QA]], %[[QB]] : tensor<8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<8x!quant.uniform<u16:f32, 1.250000e-01:32768>>) outs(%{{.*}} : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>>) -> tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>>
// CHECK-NEXT:    %[[Y:.*]] = hip.storage_cast %[[S]] : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>> to tensor<8xui16>
// CHECK-NEXT:    return %[[Y]]
// CHECK-NOT:     hip.quantize_linear
// CHECK-NOT:     hip.dequantize_linear
func.func @add(%ctx: !hip.context, %a: tensor<8xui16>, %b: tensor<8xui16>) -> tensor<8xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %db_e = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y : tensor<8xui16>
}

// -----

// The op keeps its attributes (transB) because it is retyped in place.
// CHECK-LABEL: func.func @matmul_trans_b
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<6x8x!quant.uniform<i8:f32, 1.250000e-01:3>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 5.000000e-01:32768>>) {transB = 1 : i64} : tensor<4x6x!quant.uniform<u16:f32, 5.000000e-01:32768>>
// CHECK-NOT:     hip.quantize_linear
func.func @matmul_trans_b(%ctx: !hip.context, %a: tensor<4x8xui16>, %w: tensor<6x8xi8>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %zw = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<6x8xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sb : tensor<6x8xi8>, tensor<f32>)
      zero_point(%zw : tensor<i8>)
      outs(%dw_e : tensor<6x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6x8xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<6x8xf32>) outs(%e : tensor<4x6xf32>) {transB = 1 : i64} : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xui16>
  %y = hip.quantize_linear(%ctx) ins(%m, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%za : tensor<ui16>) outs(%y_e : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %y : tensor<4x6xui16>
}

// -----

// The optional bias operand is a quantized input like any other.
// CHECK-LABEL: func.func @matmul_bias
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<8x6x!quant.uniform<i8:f32, 1.250000e-01>>, tensor<6x!quant.uniform<i32:f32, 6.250000e-02>>) outs(%{{.*}} : tensor<4x6x!quant.uniform<u16:f32, 5.000000e-01:32768>>)
// CHECK-NOT:     hip.dequantize_linear
func.func @matmul_bias(%ctx: !hip.context, %a: tensor<4x8xui16>, %w: tensor<8x6xi8>, %c: tensor<6xi32>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sc = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %zw = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %zc = hip.constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sb : tensor<8x6xi8>, tensor<f32>)
      zero_point(%zw : tensor<i8>)
      outs(%dw_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %dc_e = tensor.empty() : tensor<6xf32>
  %dc = hip.dequantize_linear(%ctx)
      ins(%c, %sc : tensor<6xi32>, tensor<f32>)
      zero_point(%zc : tensor<i32>)
      outs(%dc_e : tensor<6xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw, %dc : tensor<4x8xf32>, tensor<8x6xf32>, tensor<6xf32>) outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xui16>
  %y = hip.quantize_linear(%ctx) ins(%m, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%za : tensor<ui16>) outs(%y_e : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %y : tensor<4x6xui16>
}

// -----

// Gemm keeps alpha, beta and transB; C is a quantized input.
// CHECK-LABEL: func.func @gemm_c
// CHECK:         hip.gemm(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : {{.*}}) outs(%{{.*}} : tensor<4x6x!quant.uniform<i8:f32, 5.000000e-01>>) {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<4x6x!quant.uniform<i8:f32, 5.000000e-01>>
// CHECK-NOT:     hip.quantize_linear
func.func @gemm_c(%ctx: !hip.context, %a: tensor<4x8xi8>, %w: tensor<6x8xi8>, %c: tensor<6xi8>) -> tensor<4x6xi8> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sc = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<6x8xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sb : tensor<6x8xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%dw_e : tensor<6x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<6x8xf32>
  %dc_e = tensor.empty() : tensor<6xf32>
  %dc = hip.dequantize_linear(%ctx)
      ins(%c, %sc : tensor<6xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%dc_e : tensor<6xf32>)
      {axis = 0 : i64, block_size = 0 : i64} : tensor<6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %g = hip.gemm(%ctx) ins(%da, %dw, %dc : tensor<4x8xf32>, tensor<6x8xf32>, tensor<6xf32>) outs(%e : tensor<4x6xf32>) {alpha = 2.000000e+00 : f32, beta = 5.000000e-01 : f32, transB = 1 : i64} : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xi8>
  %y = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%z : tensor<i8>) outs(%y_e : tensor<4x6xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xi8>
  return %y : tensor<4x6xi8>
}

// -----

// CHECK-LABEL: func.func @gemm_no_c
// CHECK:         hip.gemm(%{{.*}}) ins(%{{.*}}, %{{.*}} : {{.*}}) outs(%{{.*}} : tensor<4x6x!quant.uniform<i8:f32, 5.000000e-01>>)
// CHECK-NOT:     hip.dequantize_linear
func.func @gemm_no_c(%ctx: !hip.context, %a: tensor<4x8xi8>, %w: tensor<8x6xi8>) -> tensor<4x6xi8> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sb : tensor<8x6xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%dw_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %g = hip.gemm(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<8x6xf32>) outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xi8>
  %y = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%z : tensor<i8>) outs(%y_e : tensor<4x6xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xi8>
  return %y : tensor<4x6xi8>
}

// -----

// A dynamic extent: the new init is sized from the old one.
// CHECK-LABEL: func.func @add_dynamic
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<?x8x!quant.uniform<u16:f32, 5.000000e-01:32768>>
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<?x8x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<?x8x!quant.uniform<u16:f32, 1.250000e-01:32768>>) outs(%[[E]] :
// CHECK-NOT:     hip.quantize_linear
func.func @add_dynamic(%ctx: !hip.context, %n: index, %a: tensor<?x8xui16>, %b: tensor<?x8xui16>) -> tensor<?x8xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty(%n) : tensor<?x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<?x8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<?x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x8xf32>
  %db_e = tensor.empty(%n) : tensor<?x8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<?x8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<?x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x8xf32>
  %e = tensor.empty(%n) : tensor<?x8xf32>
  %s = hip.add(%ctx) ins(%da, %db : tensor<?x8xf32>, tensor<?x8xf32>) outs(%e : tensor<?x8xf32>) -> tensor<?x8xf32>
  %y_e = tensor.empty(%n) : tensor<?x8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<?x8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<?x8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x8xui16>
  return %y : tensor<?x8xui16>
}

// -----

// A DQ with another user is kept for it; only the op's use is replaced.
// CHECK-LABEL: func.func @shared_dq
// CHECK:         hip.dequantize_linear
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8x!quant.uniform
// CHECK-NOT:     hip.quantize_linear
func.func @shared_dq(%ctx: !hip.context, %a: tensor<8xui16>, %b: tensor<8xui16>) -> (tensor<8xui16>, tensor<8xf32>) {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %db_e = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y, %da : tensor<8xui16>, tensor<8xf32>
}

// -----

// All or none: a float input next to a DQ means the op stays as it is.
// CHECK-LABEL: func.func @one_input_not_dq
// CHECK:         hip.dequantize_linear
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK:         hip.quantize_linear
// CHECK-NOT:     hip.storage_cast
func.func @one_input_not_dq(%ctx: !hip.context, %a: tensor<8xui16>, %f: tensor<8xf32>) -> tensor<8xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%da, %f : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y : tensor<8xui16>
}

// -----

// The result has a second user besides the Q.
// CHECK-LABEL: func.func @two_users
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @two_users(%ctx: !hip.context, %a: tensor<8xui16>, %b: tensor<8xui16>) -> (tensor<8xui16>, tensor<8xf32>) {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %db_e = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y, %s : tensor<8xui16>, tensor<8xf32>
}

// -----

// A per-column weight is not per-tensor: it stays for QMatMulFusion.pdll.
// CHECK-LABEL: func.func @per_column_weight
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x6xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @per_column_weight(%ctx: !hip.context, %a: tensor<4x8xui16>, %w: tensor<8x6xi8>) -> tensor<4x6xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %sw = hip.constant {value = dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]> : tensor<6xf32>} : tensor<6xf32>
  %za = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %zw = hip.constant {value = dense<[0, 0, 0, 0, 0, 0]> : tensor<6xi8>} : tensor<6xi8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %e1 = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %sw : tensor<8x6xi8>, tensor<6xf32>)
      zero_point(%zw : tensor<6xi8>) outs(%e1 : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<8x6xf32>) outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xui16>
  %y = hip.quantize_linear(%ctx) ins(%m, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%za : tensor<ui16>) outs(%y_e : tensor<4x6xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xui16>
  return %y : tensor<4x6xui16>
}

// -----

// A packed 4-bit weight is not one value per element.
// CHECK-LABEL: func.func @packed_int4
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8xf32>, tensor<8x6xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @packed_int4(%ctx: !hip.context, %a: tensor<4x8xi8>, %w: tensor<8x6xi8>) -> tensor<4x6xi8> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %da_e = tensor.empty() : tensor<4x8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<4x8xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%da_e : tensor<4x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x8xf32>
  %dw_e = tensor.empty() : tensor<8x6xf32>
  %dw = hip.dequantize_linear(%ctx)
      ins(%w, %sb : tensor<8x6xi8>, tensor<f32>)
      zero_point(%z : tensor<i8>)
      outs(%dw_e : tensor<8x6xf32>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4} : tensor<8x6xf32>
  %e = tensor.empty() : tensor<4x6xf32>
  %m = hip.matmul(%ctx) ins(%da, %dw : tensor<4x8xf32>, tensor<8x6xf32>) outs(%e : tensor<4x6xf32>) : tensor<4x6xf32>
  %y_e = tensor.empty() : tensor<4x6xi8>
  %y = hip.quantize_linear(%ctx) ins(%m, %sy : tensor<4x6xf32>, tensor<f32>)
      zero_point(%z : tensor<i8>) outs(%y_e : tensor<4x6xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x6xi8>
  return %y : tensor<4x6xi8>
}

// -----

// A scale that is not a constant cannot become a type parameter.
// CHECK-LABEL: func.func @runtime_scale
// CHECK:         hip.dequantize_linear
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @runtime_scale(%ctx: !hip.context, %a: tensor<8xui16>, %b: tensor<8xui16>, %sa: tensor<f32>) -> tensor<8xui16> {
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %db_e = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.add(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y : tensor<8xui16>
}

// -----

// hip.mul is deny-listed (it still reaches a kernel through QMulFusion.pdll), so
// it has no HipQuantCapableOpInterface and is left alone.
// CHECK-LABEL: func.func @mul_not_capable
// CHECK:         hip.mul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<8xf32>, tensor<8xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @mul_not_capable(%ctx: !hip.context, %a: tensor<8xui16>, %b: tensor<8xui16>) -> tensor<8xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sb = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %db_e = tensor.empty() : tensor<8xf32>
  %db = hip.dequantize_linear(%ctx)
      ins(%b, %sb : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%db_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %s = hip.mul(%ctx) ins(%da, %db : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) -> tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%s, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y : tensor<8xui16>
}

// -----

// hip.gelu does not declare the interface; it gets it at dialect load.
// CHECK-LABEL: func.func @gelu_attached
// CHECK:         %[[QX:.*]] = hip.storage_cast %{{.*}} : tensor<8xui16> to tensor<8x!quant.uniform<u16:f32, 2.500000e-01:32768>>
// CHECK:         %[[G:.*]] = hip.gelu(%{{.*}}) ins(%[[QX]] : tensor<8x!quant.uniform<u16:f32, 2.500000e-01:32768>>) outs(%{{.*}} : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>>)
// CHECK-NEXT:    hip.storage_cast %[[G]] : tensor<8x!quant.uniform<u16:f32, 5.000000e-01:32768>> to tensor<8xui16>
// CHECK-NOT:     hip.quantize_linear
func.func @gelu_attached(%ctx: !hip.context, %a: tensor<8xui16>) -> tensor<8xui16> {
  %sa = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %sy = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<8xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %sa : tensor<8xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8xf32>
  %e = tensor.empty() : tensor<8xf32>
  %g = hip.gelu(%ctx) ins(%da : tensor<8xf32>) outs(%e : tensor<8xf32>) : tensor<8xf32>
  %y_e = tensor.empty() : tensor<8xui16>
  %y = hip.quantize_linear(%ctx) ins(%g, %sy : tensor<8xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8xui16>
  return %y : tensor<8xui16>
}

// -----

// Integer operands (gather indices) are never quantized: only the data folds.
// CHECK-LABEL: func.func @gather_indices
// CHECK-SAME:    %[[I:[^:]*]]: tensor<2xi64>
// CHECK:         %[[QD:.*]] = hip.storage_cast %{{.*}} : tensor<3x4xui16> to tensor<3x4x!quant.uniform<u16:f32, 2.500000e-01:32768>>
// CHECK:         %[[G:.*]] = hip.gather(%{{.*}}) ins(%[[QD]], %[[I]] : tensor<3x4x!quant.uniform<u16:f32, 2.500000e-01:32768>>, tensor<2xi64>) outs(%{{.*}} : tensor<2x4x!quant.uniform<u16:f32, 2.500000e-01:32768>>)
// CHECK-NEXT:    hip.storage_cast %[[G]]
// CHECK-NOT:     hip.quantize_linear
func.func @gather_indices(%ctx: !hip.context, %d: tensor<3x4xui16>, %i: tensor<2xi64>) -> tensor<2x4xui16> {
  %s = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %dd_e = tensor.empty() : tensor<3x4xf32>
  %dd = hip.dequantize_linear(%ctx)
      ins(%d, %s : tensor<3x4xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%dd_e : tensor<3x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<3x4xf32>
  %e = tensor.empty() : tensor<2x4xf32>
  %g = hip.gather(%ctx) ins(%dd, %i : tensor<3x4xf32>, tensor<2xi64>) outs(%e : tensor<2x4xf32>) {axis = 0 : i64} : tensor<2x4xf32>
  %y_e = tensor.empty() : tensor<2x4xui16>
  %y = hip.quantize_linear(%ctx) ins(%g, %s : tensor<2x4xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<2x4xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<2x4xui16>
  return %y : tensor<2x4xui16>
}

// -----

// Layout ops are deny-listed: QdqRoundTrip runs them on storage ints instead.
// CHECK-LABEL: func.func @transpose_not_capable
// CHECK:         hip.transpose(%{{.*}}) ins(%{{.*}} : tensor<2x4xf32>)
// CHECK-NOT:     hip.storage_cast
func.func @transpose_not_capable(%ctx: !hip.context, %a: tensor<2x4xui16>) -> tensor<4x2xui16> {
  %s = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %da_e = tensor.empty() : tensor<2x4xf32>
  %da = hip.dequantize_linear(%ctx)
      ins(%a, %s : tensor<2x4xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>)
      outs(%da_e : tensor<2x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<2x4xf32>
  %e = tensor.empty() : tensor<4x2xf32>
  %t = hip.transpose(%ctx) ins(%da : tensor<2x4xf32>) outs(%e : tensor<4x2xf32>) {perm = [1, 0]} : tensor<4x2xf32>
  %y_e = tensor.empty() : tensor<4x2xui16>
  %y = hip.quantize_linear(%ctx) ins(%t, %s : tensor<4x2xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%y_e : tensor<4x2xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x2xui16>
  return %y : tensor<4x2xui16>
}
