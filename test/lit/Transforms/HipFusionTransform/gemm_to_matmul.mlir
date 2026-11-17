// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.gemm that is a plain matmul plus a row bias becomes hip.matmul. Gemms
// with alpha or beta != 1, or a C that is not a row bias, stay hip.gemm.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @row_bias
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<64x128xf16>, tensor<32x128xf16>, tensor<32xf16>) outs(%{{.*}} : tensor<64x32xf16>) {transB = 1 : i64} : tensor<64x32xf16>
// CHECK-NOT:     hip.gemm
func.func @row_bias(%ctx: !hip.context, %a: tensor<64x128xf16>,
                    %b: tensor<32x128xf16>, %c: tensor<32xf16>) -> tensor<64x32xf16> {
  %e = tensor.empty() : tensor<64x32xf16>
  %y = hip.gemm(%ctx) ins(%a, %b, %c : tensor<64x128xf16>, tensor<32x128xf16>, tensor<32xf16>)
      outs(%e : tensor<64x32xf16>) {transB = 1 : i64} : tensor<64x32xf16>
  return %y : tensor<64x32xf16>
}

// -----

// No C; transA carries over.

// CHECK-LABEL: func.func @no_bias
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<128x64xf32>, tensor<128x32xf32>) outs(%{{.*}} : tensor<64x32xf32>) {transA = 1 : i64} : tensor<64x32xf32>
// CHECK-NOT:     hip.gemm
func.func @no_bias(%ctx: !hip.context, %a: tensor<128x64xf32>,
                   %b: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %e = tensor.empty() : tensor<64x32xf32>
  %y = hip.gemm(%ctx) ins(%a, %b : tensor<128x64xf32>, tensor<128x32xf32>)
      outs(%e : tensor<64x32xf32>) {transA = 1 : i64} : tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}

// -----

// A [1, N] C collapses to the [N] bias.

// CHECK-LABEL: func.func @bias_1xn
// CHECK:         %[[C:.*]] = tensor.collapse_shape %{{.*}} {{\[\[}}0, 1]] : tensor<1x32xf32> into tensor<32xf32>
// CHECK:         hip.matmul(%{{.*}}) ins(%{{.*}}, %{{.*}}, %[[C]] :
func.func @bias_1xn(%ctx: !hip.context, %a: tensor<64x128xf32>,
                    %b: tensor<128x32xf32>, %c: tensor<1x32xf32>) -> tensor<64x32xf32> {
  %e = tensor.empty() : tensor<64x32xf32>
  %y = hip.gemm(%ctx) ins(%a, %b, %c : tensor<64x128xf32>, tensor<128x32xf32>, tensor<1x32xf32>)
      outs(%e : tensor<64x32xf32>) : tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @alpha_stays_gemm
// CHECK:         hip.gemm
// CHECK-NOT:     hip.matmul
func.func @alpha_stays_gemm(%ctx: !hip.context, %a: tensor<64x128xf32>,
                            %b: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %e = tensor.empty() : tensor<64x32xf32>
  %y = hip.gemm(%ctx) ins(%a, %b : tensor<64x128xf32>, tensor<128x32xf32>)
      outs(%e : tensor<64x32xf32>) {alpha = 2.000000e+00 : f32} : tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}

// -----

// CHECK-LABEL: func.func @beta_stays_gemm
// CHECK:         hip.gemm
// CHECK-NOT:     hip.matmul
func.func @beta_stays_gemm(%ctx: !hip.context, %a: tensor<64x128xf32>,
                           %b: tensor<128x32xf32>, %c: tensor<32xf32>) -> tensor<64x32xf32> {
  %e = tensor.empty() : tensor<64x32xf32>
  %y = hip.gemm(%ctx) ins(%a, %b, %c : tensor<64x128xf32>, tensor<128x32xf32>, tensor<32xf32>)
      outs(%e : tensor<64x32xf32>) {beta = 5.000000e-01 : f32} : tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}

// -----

// A full [M, N] C is not a row bias.

// CHECK-LABEL: func.func @full_c_stays_gemm
// CHECK:         hip.gemm
// CHECK-NOT:     hip.matmul
func.func @full_c_stays_gemm(%ctx: !hip.context, %a: tensor<64x128xf32>,
                             %b: tensor<128x32xf32>, %c: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %e = tensor.empty() : tensor<64x32xf32>
  %y = hip.gemm(%ctx) ins(%a, %b, %c : tensor<64x128xf32>, tensor<128x32xf32>, tensor<64x32xf32>)
      outs(%e : tensor<64x32xf32>) : tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}
