// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A per-tensor quantize moves back through ops that only move elements
// (insert_slice, transpose, nearest resize), down to the dequantizes and the
// empty destinations. A dequantize with the quantize's scale, zero point and
// type folds to its 8-bit input; any other dequantize is requantized alone.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s

// Concat of two inputs with the quantize's parameters: the concat runs on u8.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[A:[^:]*]]: tensor<1x4x4x4xui8>, %[[B:[^:]*]]: tensor<1x4x4x4xui8>)
// CHECK-NOT: hip.dequantize_linear
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x8x4x4xui8>
// CHECK: %[[C0:.*]] = tensor.insert_slice %[[A]] into %[[E]][0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
// CHECK: %[[C1:.*]] = tensor.insert_slice %[[B]] into %[[C0]][0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
// CHECK-NOT: hip.quantize_linear
// CHECK: return %[[C1]] : tensor<1x8x4x4xui8>
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x4xui8>,
                      %b: tensor<1x4x4x4xui8>) -> tensor<1x8x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x4x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x4xf32>
  %be = tensor.empty() : tensor<1x4x4x4xf32>
  %db = hip.dequantize_linear(%ctx) ins(%b, %s : tensor<1x4x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%be : tensor<1x4x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x4xf32>
  %e = tensor.empty() : tensor<1x8x4x4xf32>
  %c0 = tensor.insert_slice %da into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
          : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  %c1 = tensor.insert_slice %db into %c0[0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
          : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%c1, %s : tensor<1x8x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y : tensor<1x8x4x4xui8>
}

// -----

// Concat whose second input is a quantized conv output with other parameters.
// That input keeps q(dq(b)), and the requantize joins the conv kernel, which
// then returns u8 with the concat's scale.

// CHECK-LABEL: func.func @rocMlir
// CHECK-DAG: %[[S:.*]] = hip.constant {value = dense<5.000000e-01> : tensor<f32>}
// CHECK-DAG: %[[S2:.*]] = hip.constant {value = dense<2.500000e-01> : tensor<f32>}
// CHECK: hip.quant_conv
// CHECK: %[[QB:.*]] = hip.quantize_linear(%{{.*}}) ins(%{{.*}}, %[[S2]] :
// CHECK: %[[DB:.*]] = hip.dequantize_linear(%{{.*}}) ins(%[[QB]], %[[S2]] :
// CHECK: %[[RQ:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[DB]], %[[S]] :
// CHECK: return %[[RQ]] : tensor<1x2x4x4xui8>

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[A:[^:]*]]: tensor<1x2x4x4xui8>, %{{[^:]*}}: tensor<1x2x4x4xui8>)
// CHECK-NOT: hip.dequantize_linear
// CHECK: %[[K:.*]] = hip.rocmlir
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x4x4x4xui8>
// CHECK: %[[C0:.*]] = tensor.insert_slice %[[A]] into %[[E]]
// CHECK: %[[C1:.*]] = tensor.insert_slice %[[K]] into %[[C0]]
// CHECK-NOT: hip.quantize_linear
// CHECK: return %[[C1]] : tensor<1x4x4x4xui8>

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x4x4xui8>,
                      %x: tensor<1x2x4x4xui8>) -> tensor<1x4x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %s2 = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %xe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%xe : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %w = hip.constant {value = dense<1> : tensor<2x2x3x3xi8>} : tensor<2x2x3x3xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %we = tensor.empty() : tensor<2x2x3x3xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %ws : tensor<2x2x3x3xi8>, tensor<f32>)
          zero_point(%wz : tensor<i8>) outs(%we : tensor<2x2x3x3xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<2x2x3x3xf32>
  %ce = tensor.empty() : tensor<1x2x4x4xf32>
  %c = hip.conv(%ctx) ins(%dx, %dw : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>)
      outs(%ce : tensor<1x2x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xf32>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %qb = hip.quantize_linear(%ctx) ins(%c, %s2 : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %be = tensor.empty() : tensor<1x2x4x4xf32>
  %db = hip.dequantize_linear(%ctx) ins(%qb, %s2 : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%be : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %e = tensor.empty() : tensor<1x4x4x4xf32>
  %c0 = tensor.insert_slice %da into %e[0, 0, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1]
          : tensor<1x2x4x4xf32> into tensor<1x4x4x4xf32>
  %c1 = tensor.insert_slice %db into %c0[0, 2, 0, 0] [1, 2, 4, 4] [1, 1, 1, 1]
          : tensor<1x2x4x4xf32> into tensor<1x4x4x4xf32>
  %ye = tensor.empty() : tensor<1x4x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%c1, %s : tensor<1x4x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x4x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x4x4x4xui8>
  return %y : tensor<1x4x4x4xui8>
}

// -----

// Transpose of a dequantize with the quantize's parameters.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[A:[^:]*]]: tensor<1x4x4x8xui8>)
// CHECK-NOT: hip.dequantize_linear
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x8x4x4xui8>
// CHECK: %[[T:.*]] = hip.transpose(%{{.*}}) ins(%[[A]] : tensor<1x4x4x8xui8>) outs(%[[E]] : tensor<1x8x4x4xui8>)
// CHECK-NOT: hip.quantize_linear
// CHECK: return %[[T]] : tensor<1x8x4x4xui8>

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x8xui8>)
    -> tensor<1x8x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x8xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x4x4x8xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x8xf32>
  %te = tensor.empty() : tensor<1x8x4x4xf32>
  %t = hip.transpose(%ctx) ins(%da : tensor<1x4x4x8xf32>)
      outs(%te : tensor<1x8x4x4xf32>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf32>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%t, %s : tensor<1x8x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y : tensor<1x8x4x4xui8>
}

// -----

// Nearest resize of a dequantize with the quantize's parameters.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[A:[^:]*]]: tensor<1x2x4x4xui8>)
// CHECK-NOT: hip.dequantize_linear
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x2x8x8xui8>
// CHECK: %[[R:.*]] = hip.resize(%{{.*}}) ins(%[[A]] : tensor<1x2x4x4xui8>) outs(%[[E]] : tensor<1x2x8x8xui8>) {{.*}}mode = 0
// CHECK-NOT: hip.quantize_linear
// CHECK: return %[[R]] : tensor<1x2x8x8xui8>

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x4x4xui8>)
    -> tensor<1x2x8x8xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %re = tensor.empty() : tensor<1x2x8x8xf32>
  %r = hip.resize(%ctx) ins(%da : tensor<1x2x4x4xf32>)
                        outs(%re : tensor<1x2x8x8xf32>)
       {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x8xf32>
  %ye = tensor.empty() : tensor<1x2x8x8xui8>
  %y = hip.quantize_linear(%ctx) ins(%r, %s : tensor<1x2x8x8xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x2x8x8xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x8x8xui8>
  return %y : tensor<1x2x8x8xui8>
}

// -----

// A linear resize computes new values, so the quantize stays.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK: hip.resize(%{{.*}}) ins(%{{.*}} : tensor<1x2x4x4xf32>)
// CHECK: hip.quantize_linear

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x4x4xui8>)
    -> tensor<1x2x8x8xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %re = tensor.empty() : tensor<1x2x8x8xf32>
  %r = hip.resize(%ctx) ins(%da : tensor<1x2x4x4xf32>)
                        outs(%re : tensor<1x2x8x8xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x8xf32>
  %ye = tensor.empty() : tensor<1x2x8x8xui8>
  %y = hip.quantize_linear(%ctx) ins(%r, %s : tensor<1x2x8x8xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x2x8x8xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x8x8xui8>
  return %y : tensor<1x2x8x8xui8>
}

// -----

// An average pool does not implement the interface.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK: hip.pool(%{{.*}}) ins(%{{.*}} : tensor<1x2x4x4xf32>)
// CHECK: hip.quantize_linear

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x4x4xui8>)
    -> tensor<1x2x2x2xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %pe = tensor.empty() : tensor<1x2x2x2xf32>
  %p = hip.pool(%ctx) ins(%da : tensor<1x2x4x4xf32>)
                      outs(%pe : tensor<1x2x2x2xf32>)
                      {ceil_mode = 0 : i64, count_include_pad = 0 : i64,
                       dilations = [1, 1], kernel_shape = [2, 2],
                       pads = [0, 0, 0, 0], pool_mode = 0 : i64,
                       storage_order = 0 : i64, strides = [2, 2]}
       : tensor<1x2x2x2xf32>
  %ye = tensor.empty() : tensor<1x2x2x2xui8>
  %y = hip.quantize_linear(%ctx) ins(%p, %s : tensor<1x2x2x2xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x2x2x2xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x2x2xui8>
  return %y : tensor<1x2x2x2xui8>
}

// -----

// A float constant is not a valid leaf, so the whole concat stays float.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK: tensor.insert_slice %{{.*}} : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
// CHECK: tensor.insert_slice %{{.*}} : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
// CHECK: hip.quantize_linear

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x4xui8>)
    -> tensor<1x8x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x4xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x4x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x4xf32>
  %k = hip.constant {value = dense<1.0> : tensor<1x4x4x4xf32>} : tensor<1x4x4x4xf32>
  %e = tensor.empty() : tensor<1x8x4x4xf32>
  %c0 = tensor.insert_slice %da into %e[0, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
          : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  %c1 = tensor.insert_slice %k into %c0[0, 4, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
          : tensor<1x4x4x4xf32> into tensor<1x8x4x4xf32>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%c1, %s : tensor<1x8x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y : tensor<1x8x4x4xui8>
}

// -----

// The float transpose result has a second reader, so the quantize stays.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK: %[[T:.*]] = hip.transpose(%{{.*}}) ins(%{{.*}} : tensor<1x4x4x8xf32>)
// CHECK: %[[Y:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[T]],
// CHECK: return %[[Y]], %[[T]]

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x8xui8>)
    -> (tensor<1x8x4x4xui8>, tensor<1x8x4x4xf32>) {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x8xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x4x4x8xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x8xf32>
  %te = tensor.empty() : tensor<1x8x4x4xf32>
  %t = hip.transpose(%ctx) ins(%da : tensor<1x4x4x8xf32>)
      outs(%te : tensor<1x8x4x4xf32>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf32>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%t, %s : tensor<1x8x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y, %t : tensor<1x8x4x4xui8>, tensor<1x8x4x4xf32>
}

// -----

// A per-axis quantize does not commute with a transpose of its axis.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK: hip.transpose(%{{.*}}) ins(%{{.*}} : tensor<1x4x4x8xf32>)
// CHECK: hip.quantize_linear(%{{.*}}) ins(%{{.*}} : tensor<1x8x4x4xf32>, tensor<8xf32>)

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x8xui8>)
    -> tensor<1x8x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x8xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%ae : tensor<1x4x4x8xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x8xf32>
  %te = tensor.empty() : tensor<1x8x4x4xf32>
  %t = hip.transpose(%ctx) ins(%da : tensor<1x4x4x8xf32>)
      outs(%te : tensor<1x8x4x4xf32>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf32>
  %ys = hip.constant {value = dense<[0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]>
          : tensor<8xf32>} : tensor<8xf32>
  %yz = hip.constant {value = dense<128> : tensor<8xui8>} : tensor<8xui8>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%t, %ys : tensor<1x8x4x4xf32>, tensor<8xf32>)
         zero_point(%yz : tensor<8xui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y : tensor<1x8x4x4xui8>
}

// -----

// An i8 dequantize under a u8 quantize does not fold: the transpose runs on
// u8 after a requantize of the i8 input.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: %[[A:[^:]*]]: tensor<1x4x4x8xi8>)
// CHECK: %[[D:.*]] = hip.dequantize_linear(%{{.*}}) ins(%[[A]],
// CHECK: %[[Q:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[D]], {{.*}} : tensor<1x4x4x8xui8>
// CHECK: %[[T:.*]] = hip.transpose(%{{.*}}) ins(%[[Q]] : tensor<1x4x4x8xui8>)
// CHECK: return %[[T]] : tensor<1x8x4x4xui8>

func.func @main_graph(%ctx: !hip.context, %a: tensor<1x4x4x8xi8>)
    -> tensor<1x8x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %z = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x4x4x8xf32>
  %da = hip.dequantize_linear(%ctx) ins(%a, %s : tensor<1x4x4x8xi8>, tensor<f32>)
          zero_point(%az : tensor<i8>) outs(%ae : tensor<1x4x4x8xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x4x4x8xf32>
  %te = tensor.empty() : tensor<1x8x4x4xf32>
  %t = hip.transpose(%ctx) ins(%da : tensor<1x4x4x8xf32>)
      outs(%te : tensor<1x8x4x4xf32>) {perm = [0, 3, 1, 2]} : tensor<1x8x4x4xf32>
  %ye = tensor.empty() : tensor<1x8x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%t, %s : tensor<1x8x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%ye : tensor<1x8x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x8x4x4xui8>
  return %y : tensor<1x8x4x4xui8>
}
