// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.qlinear_matmul accepts static rank-2 shapes with a matching contracting
// dimension and a single scale and zero point. A mismatch is rejected before
// lowering, where the runtime call would read or write the wrong extent.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @qlinear_matmul_static
// CHECK:         hip.qlinear_matmul
func.func @qlinear_matmul_static(%ctx: !hip.context,
                                 %a: tensor<2x4xui8>,
                                 %as: tensor<f32>,
                                 %az: tensor<ui8>,
                                 %b: tensor<4x3xi8>,
                                 %bs: tensor<1xf32>,
                                 %bz: tensor<1xi8>,
                                 %ys: tensor<f32>,
                                 %yz: tensor<ui8>,
                                 %y: tensor<2x3xui8>) -> tensor<2x3xui8> {
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<2x4xui8>, tensor<f32>, tensor<ui8>,
      tensor<4x3xi8>, tensor<1xf32>, tensor<1xi8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<2x3xui8>) : tensor<2x3xui8>
  return %r : tensor<2x3xui8>
}

// -----

func.func @qlinear_matmul_k_mismatch(%ctx: !hip.context,
                                     %a: tensor<2x4xui8>,
                                     %as: tensor<f32>,
                                     %az: tensor<ui8>,
                                     %b: tensor<5x3xi8>,
                                     %bs: tensor<f32>,
                                     %bz: tensor<i8>,
                                     %ys: tensor<f32>,
                                     %yz: tensor<ui8>,
                                     %y: tensor<2x3xui8>) -> tensor<2x3xui8> {
  // expected-error @below {{b rows must equal a columns}}
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<2x4xui8>, tensor<f32>, tensor<ui8>,
      tensor<5x3xi8>, tensor<f32>, tensor<i8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<2x3xui8>) : tensor<2x3xui8>
  return %r : tensor<2x3xui8>
}

// -----

func.func @qlinear_matmul_output_mismatch(%ctx: !hip.context,
                                          %a: tensor<2x4xui8>,
                                          %as: tensor<f32>,
                                          %az: tensor<ui8>,
                                          %b: tensor<4x3xi8>,
                                          %bs: tensor<f32>,
                                          %bz: tensor<i8>,
                                          %ys: tensor<f32>,
                                          %yz: tensor<ui8>,
                                          %y: tensor<2x2xui8>) -> tensor<2x2xui8> {
  // expected-error @below {{output shape must be [M, N]}}
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<2x4xui8>, tensor<f32>, tensor<ui8>,
      tensor<4x3xi8>, tensor<f32>, tensor<i8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<2x2xui8>) : tensor<2x2xui8>
  return %r : tensor<2x2xui8>
}

// -----

func.func @qlinear_matmul_dynamic_shape(%ctx: !hip.context,
                                        %a: tensor<?x4xui8>,
                                        %as: tensor<f32>,
                                        %az: tensor<ui8>,
                                        %b: tensor<4x3xi8>,
                                        %bs: tensor<f32>,
                                        %bz: tensor<i8>,
                                        %ys: tensor<f32>,
                                        %yz: tensor<ui8>,
                                        %y: tensor<?x3xui8>) -> tensor<?x3xui8> {
  // expected-error @below {{a, b, and output must have static shapes}}
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<?x4xui8>, tensor<f32>, tensor<ui8>,
      tensor<4x3xi8>, tensor<f32>, tensor<i8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<?x3xui8>) : tensor<?x3xui8>
  return %r : tensor<?x3xui8>
}

// -----

func.func @qlinear_matmul_per_column(%ctx: !hip.context,
                                     %a: tensor<2x4xui8>,
                                     %as: tensor<f32>,
                                     %az: tensor<ui8>,
                                     %b: tensor<4x3xi8>,
                                     %bs: tensor<3xf32>,
                                     %bz: tensor<3xi8>,
                                     %ys: tensor<f32>,
                                     %yz: tensor<ui8>,
                                     %y: tensor<2x3xui8>) -> tensor<2x3xui8> {
  // expected-error @below {{b scale and zero point must be a single element}}
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<2x4xui8>, tensor<f32>, tensor<ui8>,
      tensor<4x3xi8>, tensor<3xf32>, tensor<3xi8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<2x3xui8>) : tensor<2x3xui8>
  return %r : tensor<2x3xui8>
}

// -----

func.func @qlinear_matmul_dynamic_scale(%ctx: !hip.context,
                                        %a: tensor<2x4xui8>,
                                        %as: tensor<?xf32>,
                                        %az: tensor<ui8>,
                                        %b: tensor<4x3xi8>,
                                        %bs: tensor<f32>,
                                        %bz: tensor<i8>,
                                        %ys: tensor<f32>,
                                        %yz: tensor<ui8>,
                                        %y: tensor<2x3xui8>) -> tensor<2x3xui8> {
  // expected-error @below {{a scale and zero point must be a single element}}
  %r = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
      tensor<2x4xui8>, tensor<?xf32>, tensor<ui8>,
      tensor<4x3xi8>, tensor<f32>, tensor<i8>,
      tensor<f32>, tensor<ui8>)
    outs(%y : tensor<2x3xui8>) : tensor<2x3xui8>
  return %r : tensor<2x3xui8>
}
