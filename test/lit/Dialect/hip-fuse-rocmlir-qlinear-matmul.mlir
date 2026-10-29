// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-fuse-rocmlir rewrites a QLinearMatMul with inline constant parameters
// and a zero B zero point into hip.quant_matmul, a per-column zero-point
// correction, and an f32 requantize, and outlines the chain as one rocMLIR
// kernel. Every other QLinearMatMul keeps its HIP kernel.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s

// z' = 3 - 128 = -125 and the column sums of B are [6, 4, 7], so the bias is
// 125 * [6, 4, 7]. The multipliers 0.5 * [0.5, 0.25, 0.125] / 0.125 stay per
// column.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[A:.*]]: tensor<4x8xui8>, %[[B:.*]]: tensor<8x3xi8>, %[[BIAS:.*]]: tensor<1x3xi32>, %[[M:.*]]: tensor<1x3xf32>) -> tensor<4x3xui8>
// CHECK: %[[ACC:.*]] = hip.quant_matmul(%{{.*}}) ins(%[[A]], %[[B]] : tensor<4x8xui8>, tensor<8x3xi8>)
// CHECK: %[[ADD:.*]] = hip.add(%{{.*}}) ins(%[[ACC]], %[[BIAS]] :
// CHECK: %[[F:.*]] = hip.cast(%{{.*}}) ins(%[[ADD]] : tensor<4x3xi32>) outs(%{{.*}} : tensor<4x3xf32>) {to = 1 : i64}
// CHECK: %[[S:.*]] = hip.mul(%{{.*}}) ins(%[[F]], %[[M]] :
// CHECK-DAG: %[[ONE:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[YZ:.*]] = hip.constant {value = dense<7> : tensor<ui8>}
// CHECK: hip.quantize_linear(%{{.*}}) ins(%[[S]], %[[ONE]] : {{.*}}) zero_point(%[[YZ]] : tensor<ui8>)

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.constant {value = dense<{{\[\[}}750, 500, 875]]> : tensor<1x3xi32>}
// CHECK: hip.constant {value = dense<{{\[\[}}2.000000e+00, 1.000000e+00, 5.000000e-01]]> : tensor<1x3xf32>}
// CHECK: hip.rocmlir(%{{.*}}) @rocMlir
// CHECK-NOT: hip.qlinear_matmul
func.func @main_graph(%ctx: !hip.context, %a: tensor<4x8xui8>)
    -> tensor<4x3xui8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %b = hip.constant {value = dense<[[1, 0, -1], [2, 1, 0], [0, 1, 1], [1, 1, 1],
                                    [-2, 0, 3], [0, 0, 0], [1, -1, 2], [3, 2, 1]]>
      : tensor<8x3xi8>} : tensor<8x3xi8>
  %bs = hip.constant {value = dense<[0.5, 0.25, 0.125]> : tensor<3xf32>} : tensor<3xf32>
  %bz = hip.constant {value = dense<0> : tensor<3xi8>} : tensor<3xi8>
  %ys = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<7> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<4x3xui8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<4x8xui8>, tensor<f32>, tensor<ui8>, tensor<8x3xi8>,
         tensor<3xf32>, tensor<3xi8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<4x3xui8>) : tensor<4x3xui8>
  return %y : tensor<4x3xui8>
}

// -----

// An i8 a with zero point 0 needs no correction, so B may be a graph input.
// A per-tensor multiplier is one scalar and moves into the kernel.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%{{.*}}: tensor<4x8xi8>, %{{.*}}: tensor<8x3xi8>) -> tensor<4x3xi8>
// CHECK: hip.quant_matmul
// CHECK-NOT: hip.add
// CHECK: %[[M:.*]] = hip.constant {value = dense<2.000000e+00> : tensor<f32>}
// CHECK: hip.mul(%{{.*}}) ins(%{{.*}}, %[[M]] :
// CHECK: hip.quantize_linear
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %a: tensor<4x8xi8>,
                      %b: tensor<8x3xi8>) -> tensor<4x3xi8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %bs = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e = tensor.empty() : tensor<4x3xi8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<4x8xi8>, tensor<f32>, tensor<i8>, tensor<8x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>)
         outs(%e : tensor<4x3xi8>) : tensor<4x3xi8>
  return %y : tensor<4x3xi8>
}

// -----

// A non-zero B zero point has no rocMLIR form.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_matmul
func.func @main_graph(%ctx: !hip.context, %a: tensor<4x8xi8>)
    -> tensor<4x3xi8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %b = hip.constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3xi8>
  %bs = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<1> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e = tensor.empty() : tensor<4x3xi8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<4x8xi8>, tensor<f32>, tensor<i8>, tensor<8x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>)
         outs(%e : tensor<4x3xi8>) : tensor<4x3xi8>
  return %y : tensor<4x3xi8>
}

// -----

// A ui8 a always needs a correction, which needs B as an inline constant.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_matmul
func.func @main_graph(%ctx: !hip.context, %a: tensor<4x8xui8>,
                      %b: tensor<8x3xi8>) -> tensor<4x3xui8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %bs = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %e = tensor.empty() : tensor<4x3xui8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<4x8xui8>, tensor<f32>, tensor<ui8>, tensor<8x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%e : tensor<4x3xui8>) : tensor<4x3xui8>
  return %y : tensor<4x3xui8>
}

// -----

// A scale that is not an inline constant gives no multiplier.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.qlinear_matmul
func.func @main_graph(%ctx: !hip.context, %a: tensor<4x8xi8>,
                      %ys: tensor<f32>) -> tensor<4x3xi8> {
  %as = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %az = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %b = hip.constant {value = dense<1> : tensor<8x3xi8>} : tensor<8x3xi8>
  %bs = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %bz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %yz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %e = tensor.empty() : tensor<4x3xi8>
  %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
         tensor<4x8xi8>, tensor<f32>, tensor<i8>, tensor<8x3xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>)
         outs(%e : tensor<4x3xi8>) : tensor<4x3xi8>
  return %y : tensor<4x3xi8>
}
