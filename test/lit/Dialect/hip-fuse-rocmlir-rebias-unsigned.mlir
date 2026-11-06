// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A ui8 quantized tree that reaches an integer anchor becomes i8, and every
// zero point in it drops by 128, so the anchor reads signed bytes directly.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s \
// RUN:   | hip-mlir-opt --split-input-file --convert-hip-to-tosa \
// RUN:   | FileCheck %s --check-prefix=TOSA

// A quantize that feeds a dequantized conv. The kernel reads i8, so its TOSA
// has no unsigned widen and shift before the conv.

// TOSA-LABEL: func.func @rocMlir0
// TOSA-NOT: unsigned_cast
// TOSA: tosa.conv2d

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xi8>, %{{[^:]*}}: tensor<2x2x3x3xi8>
// CHECK: hip.quant_conv(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x2x4x4xi8>, tensor<2x2x3x3xi8>)

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x2x4x4xi8>
// CHECK: %[[Z:.*]] = hip.constant {value = dense<-125> : tensor<i8>}
// CHECK: %[[Q:.*]] = hip.quantize_linear(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x2x4x4xf32>, tensor<f32>) zero_point(%[[Z]] : tensor<i8>) outs(%[[E]] : tensor<1x2x4x4xi8>)
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir0 ins(%[[Q]], {{.*}} : tensor<1x2x4x4xi8>,
// CHECK-NOT: ui8
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %xe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x2x4x4xui8>, tensor<f32>)
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
  return %c : tensor<1x2x4x4xf32>
}

// -----

// Two quantizes concatenated, then a dequantized conv: the empty destination
// and both inserts become i8.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xi8>

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[QA:.*]] = hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<i8>) {{.*}} : tensor<1x1x4x4xi8>
// CHECK: %[[QB:.*]] = hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<i8>) {{.*}} : tensor<1x1x4x4xi8>
// CHECK: %[[E:.*]] = tensor.empty() : tensor<1x2x4x4xi8>
// CHECK: %[[C0:.*]] = tensor.insert_slice %[[QA]] into %[[E]]
// CHECK: %[[C1:.*]] = tensor.insert_slice %[[QB]] into %[[C0]]{{.*}} : tensor<1x1x4x4xi8> into tensor<1x2x4x4xi8>
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir0 ins(%[[C1]],
// CHECK-NOT: ui8
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x1x4x4xf32>,
                      %b: tensor<1x1x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qae = tensor.empty() : tensor<1x1x4x4xui8>
  %qa = hip.quantize_linear(%ctx) ins(%a, %s : tensor<1x1x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qae : tensor<1x1x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x1x4x4xui8>
  %qbe = tensor.empty() : tensor<1x1x4x4xui8>
  %qb = hip.quantize_linear(%ctx) ins(%b, %s : tensor<1x1x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qbe : tensor<1x1x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x1x4x4xui8>
  %e = tensor.empty() : tensor<1x2x4x4xui8>
  %c0 = tensor.insert_slice %qa into %e[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1]
          : tensor<1x1x4x4xui8> into tensor<1x2x4x4xui8>
  %c1 = tensor.insert_slice %qb into %c0[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1]
          : tensor<1x1x4x4xui8> into tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%c1, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  return %c : tensor<1x2x4x4xf32>
}

// -----

// A nearest resize and a transpose between the quantize and the conv.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xi8>

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[Q:.*]] = hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<i8>) {{.*}} : tensor<1x2x2x2xi8>
// CHECK: %[[R:.*]] = hip.resize(%{{.*}}) ins(%[[Q]] : tensor<1x2x2x2xi8>) outs(%{{.*}} : tensor<1x2x4x4xi8>)
// CHECK: %[[T:.*]] = hip.transpose(%{{.*}}) ins(%[[R]] : tensor<1x2x4x4xi8>) outs(%{{.*}} : tensor<1x2x4x4xi8>)
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir0 ins(%[[T]],
// CHECK-NOT: ui8
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x2x2xf32>)
    -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x2x2xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x2x2xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x2x2xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x2x2xui8>
  %re = tensor.empty() : tensor<1x2x4x4xui8>
  %r = hip.resize(%ctx) ins(%q : tensor<1x2x2x2xui8>)
                        outs(%re : tensor<1x2x4x4xui8>)
       {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x4x4xui8>
  %te = tensor.empty() : tensor<1x2x4x4xui8>
  %t = hip.transpose(%ctx) ins(%r : tensor<1x2x4x4xui8>)
      outs(%te : tensor<1x2x4x4xui8>) {perm = [0, 1, 3, 2]} : tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%t, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  return %c : tensor<1x2x4x4xf32>
}

// -----

// A QLinearConv chain. The input quantize and the first conv's result become
// i8; the second conv's result is a graph output, so it stays ui8.

// The second conv's correction is 5 - (130 - 128) * 18 = -31 at an interior
// output, the same as for the ui8 input.

// CHECK-LABEL: func.func @rocMlir1
// CHECK-SAME: -> tensor<1x2x4x4xi8>
// CHECK: hip.constant {value = dense<-125> : tensor<i8>}
// CHECK: hip.quantize_linear{{.*}} : tensor<1x2x4x4xi8>
// CHECK: hip.quant_conv(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x2x4x4xi8>,
// CHECK: hip.constant {value = dense<2> : tensor<i8>}
// CHECK: return %{{.*}} : tensor<1x2x4x4xi8>

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xi8>, {{.*}} -> tensor<1x2x4x4xui8>
// CHECK: hip.constant {value = dense<3> : tensor<ui8>}
// CHECK: return %{{.*}} : tensor<1x2x4x4xui8>

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[Y1:.*]] = hip.rocmlir(%{{.*}}) @rocMlir1 {{.*}} : tensor<1x2x4x4xi8>
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}-11, -19, -19, -11], [-19, -31, -31, -19]
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir0 ins(%[[Y1]], {{.*}} : tensor<1x2x4x4xui8>
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> tensor<1x2x4x4xui8> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x3x3xi8>} : tensor<2x2x3x3xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %b = hip.constant {value = dense<[5, -5]> : tensor<2xi32>} : tensor<2xi32>
  %y1z = hip.constant {value = dense<130> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %y1e = tensor.empty() : tensor<1x2x4x4xui8>
  %y1 = hip.qlinear_conv(%ctx) ins(%q, %s, %z, %w, %ws, %wz, %s, %y1z, %b :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%y1e : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xui8>
  %y2e = tensor.empty() : tensor<1x2x4x4xui8>
  %y2 = hip.qlinear_conv(%ctx) ins(%y1, %s, %y1z, %w, %ws, %wz, %s, %z, %b :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%y2e : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xui8>
  return %y2 : tensor<1x2x4x4xui8>
}

// -----

// Declined: the ui8 data is a function argument.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: 4x4xi8>
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  return %c : tensor<1x2x4x4xf32>
}

// -----

// Declined: the quantize is also a graph output.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<ui8>) {{.*}} : tensor<1x2x4x4xui8>
// CHECK: return %{{.*}}, %{{.*}} : tensor<1x2x4x4xf32>, tensor<1x2x4x4xui8>
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xui8>) {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  return %c, %q : tensor<1x2x4x4xf32>, tensor<1x2x4x4xui8>
}

// -----

// Declined: a hip.qadd also reads the quantized data.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quantize_linear{{.*}} : tensor<1x2x4x4xui8>
// CHECK: hip.qadd(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x2x4x4xui8>, tensor<1x2x4x4xui8>)
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xui8>) {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  %ae = tensor.empty() : tensor<1x2x4x4xui8>
  %add = hip.qadd(%ctx) ins(%q, %q : tensor<1x2x4x4xui8>, tensor<1x2x4x4xui8>)
           outs(%ae : tensor<1x2x4x4xui8>)
           {lhs_scale = 5.000000e-01 : f32, lhs_zp = 3 : i64,
            output_scale = 5.000000e-01 : f32, output_zp = 3 : i64,
            rhs_scale = 5.000000e-01 : f32, rhs_zp = 3 : i64} : tensor<1x2x4x4xui8>
  return %c, %add : tensor<1x2x4x4xf32>, tensor<1x2x4x4xui8>
}

// -----

// Declined: the quantize has a per-channel zero point.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: %{{[^:]*}}: tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<2xui8>) {{.*}} : tensor<1x2x4x4xui8>
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %s2 = hip.constant {value = dense<0.5> : tensor<2xf32>} : tensor<2xf32>
  %z2 = hip.constant {value = dense<[3, 4]> : tensor<2xui8>} : tensor<2xui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s2 : tensor<1x2x4x4xf32>, tensor<2xf32>)
         zero_point(%z2 : tensor<2xui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
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
  return %c : tensor<1x2x4x4xf32>
}

// -----

// Declined: the tree reaches no integer anchor.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.quantize_linear{{.*}} zero_point(%{{.*}} : tensor<ui8>) {{.*}} : tensor<1x2x4x4xui8>
// CHECK: hip.dequantize_linear{{.*}} zero_point(%{{.*}} : tensor<ui8>)
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>)
    -> tensor<1x2x4x4xf32> {
  %s = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x4x4xui8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z : tensor<ui8>) outs(%qe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  %dxe = tensor.empty() : tensor<1x2x4x4xf32>
  %dx = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z : tensor<ui8>) outs(%dxe : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  return %dx : tensor<1x2x4x4xf32>
}
