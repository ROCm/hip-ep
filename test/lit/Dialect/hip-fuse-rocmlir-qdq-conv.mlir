// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A conv of a per-tensor dequantized input and dequantized i8 weights becomes
// hip.quant_conv on the integer operands. The kernel adds the zero-point
// correction in i32, scales by x_scale * w_scale and adds the bias in f32, so
// the ops that read the conv run in the same kernel.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s

// u8 input with zero point 128, so the rebias needs no correction. The bias
// dequantizes to [2, -4] * 0.125, and the scale is 0.5 * 0.25.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: %[[X:[^:]*]]: tensor<1x2x4x4xui8>, %[[W:[^:]*]]: tensor<2x2x3x3xi8>, %[[B:[^:]*]]: tensor<1x2x1x1xf32>)
// CHECK-SAME: -> tensor<1x2x4x4xui8>
// CHECK: %[[ACC:.*]] = hip.quant_conv(%{{.*}}) ins(%[[X]], %[[W]] :
// CHECK-NOT: hip.add
// CHECK: %[[F:.*]] = hip.cast(%{{.*}}) ins(%[[ACC]] :
// CHECK: %[[M:.*]] = hip.constant {value = dense<1.250000e-01> : tensor<f32>}
// CHECK: %[[S:.*]] = hip.mul(%{{.*}}) ins(%[[F]], %[[M]] :
// CHECK: %[[C:.*]] = hip.add(%{{.*}}) ins(%[[S]], %[[B]] :
// CHECK: %[[L:.*]] = hip.leaky_relu(%{{.*}}) ins(%[[C]] :
// CHECK: %[[OUT:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[L]],
// CHECK: return %[[OUT]] : tensor<1x2x4x4xui8>

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.conv
// CHECK-NOT: hip.dequantize_linear
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}2.500000e-01]], {{\[\[}}-5.000000e-01]]]]> : tensor<1x2x1x1xf32>}
// CHECK: hip.rocmlir
// CHECK-NOT: hip.quantize_linear
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %xe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%x, %xs : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%xz : tensor<ui8>) outs(%xe : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %w = hip.constant {value = dense<1> : tensor<2x2x3x3xi8>} : tensor<2x2x3x3xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %we = tensor.empty() : tensor<2x2x3x3xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %ws : tensor<2x2x3x3xi8>, tensor<f32>)
          zero_point(%wz : tensor<i8>) outs(%we : tensor<2x2x3x3xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<2x2x3x3xf32>
  %b = hip.constant {value = dense<[2, -4]> : tensor<2xi8>} : tensor<2xi8>
  %bs = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %be = tensor.empty() : tensor<2xf32>
  %db = hip.dequantize_linear(%ctx) ins(%b, %bs : tensor<2xi8>, tensor<f32>)
          zero_point(%bz : tensor<i8>) outs(%be : tensor<2xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<2xf32>
  %ce = tensor.empty() : tensor<1x2x4x4xf32>
  %c = hip.conv(%ctx) ins(%dx, %dw, %db : tensor<1x2x4x4xf32>,
                                          tensor<2x2x3x3xf32>, tensor<2xf32>)
      outs(%ce : tensor<1x2x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xf32>
  %le = tensor.empty() : tensor<1x2x4x4xf32>
  %l = hip.leaky_relu(%ctx) ins(%c : tensor<1x2x4x4xf32>)
         outs(%le : tensor<1x2x4x4xf32>) {alpha = 0.19921875 : f64}
         : tensor<1x2x4x4xf32>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.quantize_linear(%ctx) ins(%l, %ys : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%yz : tensor<ui8>) outs(%ye : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  return %y : tensor<1x2x4x4xui8>
}

// -----

// i8 input with zero point 3 and per-channel weight scales [0.25, 0.5]. Each
// output reads both unit weights, so the correction is -3 * 2 per channel, and
// the scale is 0.5 * [0.25, 0.5]. The conv has no bias.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: %[[Z:[^:]*]]: tensor<1x2x1x1xi32>, %[[M:[^:]*]]: tensor<1x2x1x1xf32>)
// CHECK: %[[ACC:.*]] = hip.quant_conv
// CHECK: %[[A:.*]] = hip.add(%{{.*}}) ins(%[[ACC]], %[[Z]] :
// CHECK: %[[F:.*]] = hip.cast(%{{.*}}) ins(%[[A]] :
// CHECK: %[[S:.*]] = hip.mul(%{{.*}}) ins(%[[F]], %[[M]] :
// CHECK: hip.quantize_linear(%{{.*}}) ins(%[[S]],

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.conv
// CHECK: hip.constant {value = dense<-6> : tensor<1x2x1x1xi32>}
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}1.250000e-01]], {{\[\[}}2.500000e-01]]]]> : tensor<1x2x1x1xf32>}
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xi8>)
    -> tensor<1x2x4x4xi8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<3> : tensor<i8>} : tensor<i8>
  %xe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%x, %xs : tensor<1x2x4x4xi8>, tensor<f32>)
          zero_point(%xz : tensor<i8>) outs(%xe : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<[0.25, 0.5]> : tensor<2xf32>} : tensor<2xf32>
  %we = tensor.empty() : tensor<2x2x1x1xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %ws : tensor<2x2x1x1xi8>, tensor<2xf32>)
          outs(%we : tensor<2x2x1x1xf32>)
          {axis = 0 : i64, block_size = 0 : i64} : tensor<2x2x1x1xf32>
  %ce = tensor.empty() : tensor<1x2x4x4xf32>
  %c = hip.conv(%ctx) ins(%dx, %dw : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
      outs(%ce : tensor<1x2x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xf32>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ye = tensor.empty() : tensor<1x2x4x4xi8>
  %y = hip.quantize_linear(%ctx) ins(%c, %ys : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%yz : tensor<i8>) outs(%ye : tensor<1x2x4x4xi8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xi8>
  return %y : tensor<1x2x4x4xi8>
}

// -----

// Weight scales along axis 1 are not per output channel, so the conv stays a
// float conv of the dequantized operands.

// CHECK-LABEL: func.func @rocMlir
// CHECK-NOT: hip.quant_conv
// CHECK: hip.conv

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xf32> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %xe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%x, %xs : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%xz : tensor<ui8>) outs(%xe : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<[0.25, 0.5]> : tensor<2xf32>} : tensor<2xf32>
  %we = tensor.empty() : tensor<2x2x1x1xf32>
  %dw = hip.dequantize_linear(%ctx) ins(%w, %ws : tensor<2x2x1x1xi8>, tensor<2xf32>)
          outs(%we : tensor<2x2x1x1xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<2x2x1x1xf32>
  %ce = tensor.empty() : tensor<1x2x4x4xf32>
  %c = hip.conv(%ctx) ins(%dx, %dw : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
      outs(%ce : tensor<1x2x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xf32>
  return %c : tensor<1x2x4x4xf32>
}
