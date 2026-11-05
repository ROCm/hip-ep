// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.resize's attributes are ONNX Resize's: mode (nearest, linear, cubic),
// antialias, exclude_outside and cubic_coeff_a. The verifier admits the
// combinations the kernel computes: antialias on linear with exclude_outside
// and a half-pixel transform, cubic with a = -0.75 and without
// exclude_outside, both on float types.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @antialias
// CHECK: hip.resize
// CHECK-SAME: antialias = true
// CHECK-SAME: exclude_outside = true
func.func @antialias(%ctx: !hip.context, %x: tensor<1x3x32x32xf32>,
                     %init: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x32x32xf32>)
       outs(%init : tensor<1x3x16x16xf32>)
       {mode = 1 : i64, coord_transform = 3 : i64, nearest_mode = 0 : i64,
        antialias = true, exclude_outside = true} : tensor<1x3x16x16xf32>
  return %r : tensor<1x3x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @cubic
// CHECK: hip.resize
// CHECK-SAME: mode = 2
func.func @cubic(%ctx: !hip.context, %x: tensor<1x3x16x16xf16>,
                 %init: tensor<1x3x40x40xf16>) -> tensor<1x3x40x40xf16> {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf16>)
       outs(%init : tensor<1x3x40x40xf16>)
       {mode = 2 : i64, coord_transform = 2 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x40x40xf16>
  return %r : tensor<1x3x40x40xf16>
}

// -----

func.func @antialias_nearest(%ctx: !hip.context, %x: tensor<1x3x32x32xf32>,
                             %init: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
  // expected-error @+1 {{antialias is supported for linear only}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x32x32xf32>)
       outs(%init : tensor<1x3x16x16xf32>)
       {mode = 0 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64,
        antialias = true, exclude_outside = true} : tensor<1x3x16x16xf32>
  return %r : tensor<1x3x16x16xf32>
}

// -----

func.func @antialias_without_exclude(%ctx: !hip.context, %x: tensor<1x3x32x32xf32>,
                                     %init: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
  // expected-error @+1 {{antialias requires exclude_outside}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x32x32xf32>)
       outs(%init : tensor<1x3x16x16xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64,
        antialias = true} : tensor<1x3x16x16xf32>
  return %r : tensor<1x3x16x16xf32>
}

// -----

func.func @antialias_align_corners(%ctx: !hip.context, %x: tensor<1x3x32x32xf32>,
                                   %init: tensor<1x3x16x16xf32>) -> tensor<1x3x16x16xf32> {
  // expected-error @+1 {{antialias requires half_pixel or pytorch_half_pixel coordinates}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x32x32xf32>)
       outs(%init : tensor<1x3x16x16xf32>)
       {mode = 1 : i64, coord_transform = 2 : i64, nearest_mode = 0 : i64,
        antialias = true, exclude_outside = true} : tensor<1x3x16x16xf32>
  return %r : tensor<1x3x16x16xf32>
}

// -----

func.func @cubic_integer(%ctx: !hip.context, %x: tensor<1x3x16x16xui8>,
                         %init: tensor<1x3x40x40xui8>) -> tensor<1x3x40x40xui8> {
  // expected-error @+1 {{cubic requires a float element type}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xui8>)
       outs(%init : tensor<1x3x40x40xui8>)
       {mode = 2 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x40x40xui8>
  return %r : tensor<1x3x40x40xui8>
}

// -----

func.func @cubic_exclude_outside(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                                 %init: tensor<1x3x40x40xf32>) -> tensor<1x3x40x40xf32> {
  // expected-error @+1 {{cubic with exclude_outside is not supported}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
       outs(%init : tensor<1x3x40x40xf32>)
       {mode = 2 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64,
        exclude_outside = true} : tensor<1x3x40x40xf32>
  return %r : tensor<1x3x40x40xf32>
}

// -----

func.func @cubic_coeff(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                       %init: tensor<1x3x40x40xf32>) -> tensor<1x3x40x40xf32> {
  // expected-error @+1 {{cubic supports cubic_coeff_a = -0.75 only}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
       outs(%init : tensor<1x3x40x40xf32>)
       {mode = 2 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64,
        cubic_coeff_a = -5.000000e-01 : f32} : tensor<1x3x40x40xf32>
  return %r : tensor<1x3x40x40xf32>
}

// -----

func.func @bad_mode(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                    %init: tensor<1x3x40x40xf32>) -> tensor<1x3x40x40xf32> {
  // expected-error @+1 {{mode must be 0 (nearest), 1 (linear) or 2 (cubic)}}
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
       outs(%init : tensor<1x3x40x40xf32>)
       {mode = 3 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x40x40xf32>
  return %r : tensor<1x3x40x40xf32>
}
