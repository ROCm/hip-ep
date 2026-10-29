// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-fuse-rocmlir rewrites a QLinearConv with inline constants and a zero
// weight zero point into hip.quant_conv, a constant zero-point correction,
// and an f32 requantize, and outlines the chain as one rocMLIR kernel. Every
// other QLinearConv keeps its HIP kernel.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s

// Padding: the correction -z' * sum(w) varies over the output plane. With
// z' = 3 - 128 = -125, channel 0 reads 12 weight units inside and 6 at a
// corner, so its bias is 5 + 125 * 12 = 1505 and 5 + 125 * 6 = 755. The
// multipliers 0.5 * [0.5, 0.25] / 0.125 = [2, 1] stay per channel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[X:.*]]: tensor<1x2x4x4xui8>, %[[W:.*]]: tensor<2x2x3x3xi8>, %[[B:.*]]: tensor<1x2x4x4xi32>, %[[M:.*]]: tensor<1x2x1x1xf32>) -> tensor<1x2x4x4xui8>
// CHECK: %[[ACC:.*]] = hip.quant_conv(%{{.*}}) ins(%[[X]], %[[W]] :
// CHECK-SAME: pads = [1, 1, 1, 1]
// CHECK: %[[ADD:.*]] = hip.add(%{{.*}}) ins(%[[ACC]], %[[B]] :
// CHECK: %[[F:.*]] = hip.cast(%{{.*}}) ins(%[[ADD]] : tensor<1x2x4x4xi32>) outs(%{{.*}} : tensor<1x2x4x4xf32>) {to = 1 : i64}
// CHECK: %[[S:.*]] = hip.mul(%{{.*}}) ins(%[[F]], %[[M]] :
// CHECK-DAG: %[[ONE:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[YZ:.*]] = hip.constant {value = dense<7> : tensor<ui8>}
// CHECK: hip.quantize_linear(%{{.*}}) ins(%[[S]], %[[ONE]] : {{.*}}) zero_point(%[[YZ]] : tensor<ui8>)
// CHECK-SAME: saturate = 1

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}755, 1005, 1005, 630], [1005, 1505, 1505, 1005], [1005, 1505, 1505, 1005], [630, 1005, 1005, 755]], {{\[\[}}370, 370, 370, 370], [370, 245, 245, 245], [370, 245, 245, 245], [370, 245, 245, 245]]]]> : tensor<1x2x4x4xi32>}
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}2.000000e+00]], {{\[\[}}1.000000e+00]]]]> : tensor<1x2x1x1xf32>}
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir
// CHECK-NOT: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<[[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                     [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                                    [[[-1, 0, 0], [0, 2, 0], [0, 0, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]>
      : tensor<2x2x3x3xi8>} : tensor<2x2x3x3xi8>
  %ws = hip.constant {value = dense<[0.5, 0.25]> : tensor<2xf32>} : tensor<2xf32>
  %wz = hip.constant {value = dense<0> : tensor<2xi8>} : tensor<2xi8>
  %ys = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<7> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[5, -5]> : tensor<2xi32>} : tensor<2xi32>
  %e = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x3x3xi8>,
         tensor<2xf32>, tensor<2xi8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%e : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xui8>
  return %y : tensor<1x2x4x4xui8>
}

// -----

// No padding: every output reads every tap, so the correction folds into a
// per-channel bias, B + 128 * sum(w) = [10 + 384, 20 - 128]. A per-tensor
// multiplier is one scalar and moves into the kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[X:.*]]: tensor<1x2x3x3xui8>, %[[W:.*]]: tensor<2x2x1x1xi8>, %[[B:.*]]: tensor<1x2x1x1xi32>) -> tensor<1x2x3x3xui8>
// CHECK: hip.quant_conv
// CHECK: hip.add(%{{.*}}) ins(%{{.*}}, %[[B]] :
// CHECK: %[[M:.*]] = hip.constant {value = dense<2.000000e+00> : tensor<f32>}
// CHECK: hip.mul(%{{.*}}) ins(%{{.*}}, %[[M]] :
// CHECK: hip.quantize_linear

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.constant {value = dense<{{\[\[\[\[}}394]], {{\[\[}}-108]]]]> : tensor<1x2x1x1xi32>}
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<[[[[1]], [[2]]], [[[3]], [[-4]]]]>
      : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[10, 20]> : tensor<2xi32>} : tensor<2xi32>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// An i8 input with zero point 0 and no bias needs no correction at all.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{.*}}: tensor<1x2x3x3xi8>, %{{.*}}: tensor<2x2x1x1xi8>) -> tensor<1x2x3x3xi8>
// CHECK: hip.quant_conv
// CHECK-NOT: hip.add
// CHECK: hip.cast
// CHECK: hip.quantize_linear
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xi8>)
    -> tensor<1x2x3x3xi8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e = tensor.empty() : tensor<1x2x3x3xi8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xi8>, tensor<f32>, tensor<i8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>)
         outs(%e : tensor<1x2x3x3xi8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xi8>
  return %y : tensor<1x2x3x3xi8>
}

// -----

// A QuantizeLinear that feeds the conv joins the kernel as its prologue, so
// the kernel reads the f32 activation directly.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{.*}}: tensor<1x2x3x3xf32>,
// CHECK: hip.quantize_linear
// CHECK: hip.quant_conv
// CHECK: hip.quantize_linear
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.quantize_linear
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %a: tensor<1x2x3x3xf32>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %qe = tensor.empty() : tensor<1x2x3x3xui8>
  %x = hip.quantize_linear(%ctx) ins(%a, %xs : tensor<1x2x3x3xf32>, tensor<f32>)
         zero_point(%xz : tensor<ui8>) outs(%qe : tensor<1x2x3x3xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x3x3xui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// A non-zero weight zero point has no rocMLIR form.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<1> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// Weights that are not an inline constant cannot be summed for the
// correction.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>,
                      %w: tensor<2x2x1x1xi8>) -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// A scale that is not an inline constant gives no multiplier.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>,
                      %xs: tensor<f32>) -> tensor<1x2x3x3xui8> {
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// A dynamic batch keeps the HIP kernel.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<?x2x3x3xui8>,
                      %e: tensor<?x2x3x3xui8>) -> tensor<?x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<?x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<?x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<?x2x3x3xui8>
  return %y : tensor<?x2x3x3xui8>
}

// -----

// A correction outside the i32 range keeps the HIP kernel: with z' = -128,
// a bias of 2^31 - 1 plus 128 * 2 overflows.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<2147483647> : tensor<2xi32>} : tensor<2xi32>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}
