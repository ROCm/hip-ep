// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// hip-rocmlir-prepare-quant rewrites a QLinearConv or QLinearMatMul with ui8
// weights or a non-zero weight zero point into integer ops on i8 weights
// w' = w - offset, with zw' = zw - offset:
//
//   sum (x' - z')(w' - zw') = quant(x, w') - quant(x, zw') - z' * sum(w')
//                             + z' * zw' * taps
//
// The second integer op runs over a filter of zw' in the weights' shape; the
// constant terms join the i32 bias. A channel whose requantize multiplier is
// below 2^-33 always writes the output zero point, so its bias and its
// multiplier are 0 and its correction cannot overflow.
//
// 1. ui8 weights, zw = 128: zw' = 0, one quant_conv on w - 128.
// 2. ui8 weights, zw = 130, z' = -125, a bias: two quant_convs, the bias
//    B - z' * sum(w') + z' * zw' * taps.
// 3. i8 weights, zw = 2, padding: the correction varies over the plane.
// 4. A dead channel: an INT32_MIN bias and a 1e-15 weight scale.
// 5. QLinearMatMul, ui8 B with per-column zero points [128, 130].
// 6. A dynamic batch keeps the HIP kernel.
// ============================================================================

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant %s \
// RUN:   | FileCheck %s --check-prefix=SPLIT
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | FileCheck %s --check-prefix=FUSE

// SPLIT-LABEL: func.func @main_graph
// SPLIT: %[[W:.*]] = hip.constant {value = dense<{{\[\[\[\[}}2]], {{\[\[}}1]]], {{\[\[\[}}-1]], {{\[\[}}0]]]]> : tensor<2x2x1x1xi8>}
// SPLIT: hip.quant_conv(%{{.*}}) ins(%{{.*}}, %[[W]] :
// SPLIT-NOT: hip.quant_conv
// SPLIT-NOT: hip.sub
// SPLIT: hip.quantize_linear
// SPLIT-NOT: hip.qlinear_conv

// One kernel; z' = -128, so the bias is 128 * sum(w') = [384, -128].
// FUSE-LABEL: func.func @main_graph
// FUSE: hip.constant {value = dense<{{\[\[\[\[}}384]], {{\[\[}}-128]]]]> : tensor<1x2x1x1xi32>}
// FUSE: hip.rocmlir(
// FUSE-NOT: hip.rocmlir(
// FUSE-NOT: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<[[[[130]], [[129]]], [[[127]], [[128]]]]>
      : tensor<2x2x1x1xui8>} : tensor<2x2x1x1xui8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xui8>,
         tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// w' = [[3, 1], [-2, 2]], zw' = 2, z' = 3 - 128 = -125, taps = 2:
// bias = [10, -20] + 125 * [4, 0] - 125 * 2 * 2 = [10, -520].

// SPLIT-LABEL: func.func @main_graph
// SPLIT: %[[W:.*]] = hip.constant {value = dense<{{\[\[\[\[}}3]], {{\[\[}}1]]], {{\[\[\[}}-2]], {{\[\[}}2]]]]> : tensor<2x2x1x1xi8>}
// SPLIT: %[[ACC:.*]] = hip.quant_conv(%{{.*}}) ins(%{{.*}}, %[[W]] :
// SPLIT: %[[Z:.*]] = hip.constant {value = dense<2> : tensor<2x2x1x1xi8>}
// SPLIT: %[[SUM:.*]] = hip.quant_conv(%{{.*}}) ins(%{{.*}}, %[[Z]] :
// SPLIT: %[[D:.*]] = hip.sub(%{{.*}}) ins(%[[ACC]], %[[SUM]] :
// SPLIT: %[[B:.*]] = hip.constant {value = dense<{{\[\[\[\[}}10]], {{\[\[}}-520]]]]> : tensor<1x2x1x1xi32>}
// SPLIT: hip.add(%{{.*}}) ins(%[[D]], %[[B]] :
// SPLIT: hip.cast
// SPLIT: %[[M:.*]] = hip.constant {value = dense<2.000000e+00> : tensor<f32>}
// SPLIT: hip.mul(%{{.*}}) ins(%{{.*}}, %[[M]] :
// SPLIT: hip.quantize_linear
// SPLIT-NOT: hip.qlinear_conv

// The window-sum conv is a kernel of its own; the main kernel's epilogue
// subtracts its result before the bias and the requantize.
// FUSE-LABEL: func.func @rocMlir1(
// FUSE: hip.quant_conv
// FUSE: hip.sub
// FUSE: hip.add
// FUSE: hip.quantize_linear
// FUSE-LABEL: func.func @rocMlir0(
// FUSE-NOT: hip.sub
// FUSE: hip.quant_conv
// FUSE-LABEL: func.func @main_graph
// FUSE: %[[SUM:.*]] = hip.rocmlir(%{{.*}}) @rocMlir0
// FUSE: hip.constant {value = dense<{{\[\[\[\[}}10]], {{\[\[}}-520]]]]> : tensor<1x2x1x1xi32>}
// FUSE: hip.rocmlir(%{{.*}}) @rocMlir1 ins(%{{.*}}, %{{.*}}, %[[SUM]], %{{.*}} :
// FUSE-NOT: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<[[[[131]], [[129]]], [[[126]], [[130]]]]>
      : tensor<2x2x1x1xui8>} : tensor<2x2x1x1xui8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<130> : tensor<ui8>} : tensor<ui8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[10, -20]> : tensor<2xi32>} : tensor<2xi32>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xui8>,
         tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// A 2x2 input, a 3x3 kernel of ones and pads of 1: every output reads 4 valid
// taps, but the taps differ by position, so the bias keeps the plane. With
// z' = -125 and zw' = 2 it is 125 * 4 - 125 * 2 * 4 = -500 everywhere.

// SPLIT-LABEL: func.func @main_graph
// SPLIT: hip.quant_conv
// SPLIT: hip.constant {value = dense<2> : tensor<1x1x3x3xi8>}
// SPLIT: hip.quant_conv
// SPLIT: hip.sub
// SPLIT: hip.constant {value = dense<-500> : tensor<1x1x2x2xi32>}

// FUSE-LABEL: func.func @main_graph
// FUSE: hip.rocmlir(%{{.*}}) @rocMlir0
// FUSE: hip.constant {value = dense<-500> : tensor<1x1x2x2xi32>}
// FUSE: hip.rocmlir(%{{.*}}) @rocMlir1
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x1x2x2xui8>)
    -> tensor<1x1x2x2xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<1x1x3x3xi8>} : tensor<1x1x3x3xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<2> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<1x1x2x2xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>, tensor<1x1x3x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<1x1x2x2xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x1x2x2xui8>
  return %y : tensor<1x1x2x2xui8>
}

// -----

// Channel 0 has the bias INT32_MIN and the multiplier 0.5 * 1e-15 / 0.0625,
// below 2^-33: its correction INT32_MIN - 256 would overflow, but the channel
// writes the zero point for any accumulator, so it gets bias 0 and multiplier
// 0. Channel 1 is 5 + 128 * 2 = 261 with multiplier 2. The plan is the usual
// one, so the anchor fusion decomposes the conv into one kernel.

// SPLIT-LABEL: func.func @main_graph
// SPLIT: hip.qlinear_conv

// FUSE-LABEL: func.func @main_graph
// FUSE-DAG: hip.constant {value = dense<{{\[\[\[\[}}0]], {{\[\[}}261]]]]> : tensor<1x2x1x1xi32>}
// FUSE-DAG: hip.constant {value = dense<{{\[\[\[\[}}0.000000e+00]], {{\[\[}}2.000000e+00]]]]> : tensor<1x2x1x1xf32>}
// FUSE: hip.rocmlir(
// FUSE-NOT: hip.rocmlir(
// FUSE-NOT: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x3x3xui8>)
    -> tensor<1x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<[[[[-1]], [[-1]]], [[[1]], [[1]]]]>
      : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<[1.0e-15, 0.25]> : tensor<2xf32>} : tensor<2xf32>
  %wz = hip.constant {value = dense<0> : tensor<2xi8>} : tensor<2xi8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[-2147483648, 5]> : tensor<2xi32>} : tensor<2xi32>
  %e = tensor.empty() : tensor<1x2x3x3xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
         tensor<1x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<2xf32>, tensor<2xi8>, tensor<f32>, tensor<ui8>, tensor<2xi32>)
         outs(%e : tensor<1x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x3x3xui8>
  return %y : tensor<1x2x3x3xui8>
}

// -----

// B' = B - 128 = [[1, 3], [-1, 2], [0, 0]], zb' = [0, 2], z' = 5 - 128 = -123,
// K = 3: bias = 123 * [0, 5] - 123 * [0, 2] * 3 = [0, -123].

// SPLIT-LABEL: func.func @main_graph
// SPLIT: %[[B:.*]] = hip.constant {value = dense<{{\[\[}}1, 3], [-1, 2], [0, 0]]> : tensor<3x2xi8>}
// SPLIT: %[[ACC:.*]] = hip.quant_matmul(%{{.*}}) ins(%{{.*}}, %[[B]] :
// SPLIT: %[[Z:.*]] = hip.constant {value = dense<{{\[\[}}0, 2], [0, 2], [0, 2]]> : tensor<3x2xi8>}
// SPLIT: %[[SUM:.*]] = hip.quant_matmul(%{{.*}}) ins(%{{.*}}, %[[Z]] :
// SPLIT: hip.sub(%{{.*}}) ins(%[[ACC]], %[[SUM]] :
// SPLIT: hip.constant {value = dense<{{\[\[}}0, -123]]> : tensor<1x2xi32>}
// SPLIT-NOT: hip.qlinear_matmul

// FUSE-LABEL: func.func @rocMlir1(
// FUSE: hip.quant_matmul
// FUSE: hip.sub
// FUSE-LABEL: func.func @main_graph
// FUSE: hip.rocmlir(%{{.*}}) @rocMlir0
// FUSE: hip.constant {value = dense<{{\[\[}}0, -123]]> : tensor<1x2xi32>}
// FUSE: hip.rocmlir(%{{.*}}) @rocMlir1
// FUSE-NOT: hip.qlinear_matmul
func.func @main_graph(%ctx: !hip.context, %a: tensor<2x3xui8>)
    -> tensor<2x2xui8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<5> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[[129, 131], [127, 130], [128, 128]]>
      : tensor<3x2xui8>} : tensor<3x2xui8>
  %bs = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<[128, 130]> : tensor<2xui8>} : tensor<2xui8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<2x2xui8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<2x3xui8>, tensor<f32>, tensor<ui8>, tensor<3x2xui8>,
         tensor<f32>, tensor<2xui8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<2x2xui8>) : tensor<2x2xui8>
  return %y : tensor<2x2xui8>
}

// -----

// A dynamic batch keeps the HIP kernel.

// SPLIT-LABEL: func.func @main_graph
// SPLIT: hip.qlinear_conv
// FUSE-LABEL: func.func @main_graph
// FUSE-NOT: hip.rocmlir(
// FUSE: hip.qlinear_conv
func.func @main_graph(%ctx: !hip.context, %x: tensor<?x2x3x3xui8>,
                      %e: tensor<?x2x3x3xui8>) -> tensor<?x2x3x3xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<130> : tensor<2x2x1x1xui8>} : tensor<2x2x1x1xui8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<130> : tensor<ui8>} : tensor<ui8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<?x2x3x3xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xui8>,
         tensor<f32>, tensor<ui8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<?x2x3x3xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<?x2x3x3xui8>
  return %y : tensor<?x2x3x3xui8>
}
