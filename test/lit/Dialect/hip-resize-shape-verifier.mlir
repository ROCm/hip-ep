// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics %s

func.func @dynamic_input_static_output(
    %ctx: !hip.context,
    %input: tensor<?x3x16x16xf16>,
    %output: tensor<2x3x32x32xf16>) {
  // expected-error @+1 {{'hip.resize' op resize output dimension 0 must remain dynamic because input N/C is dynamic}}
  %result = hip.resize(%ctx)
      ins(%input : tensor<?x3x16x16xf16>)
      outs(%output : tensor<2x3x32x32xf16>)
      {mode = 1, coord_transform = 0, nearest_mode = 0}
      : tensor<2x3x32x32xf16>
  return
}

func.func @dynamic_spatial_output(
    %ctx: !hip.context,
    %input: tensor<1x3x16x16xf16>,
    %output: tensor<1x3x?x32xf16>) {
  // expected-error @+1 {{'hip.resize' op resize output spatial dimension 2 must be static because sizes/scales are not carried by hip.resize}}
  %result = hip.resize(%ctx)
      ins(%input : tensor<1x3x16x16xf16>)
      outs(%output : tensor<1x3x?x32xf16>)
      {mode = 0, coord_transform = 1, nearest_mode = 2}
      : tensor<1x3x?x32xf16>
  return
}

func.func @mismatched_channels(
    %ctx: !hip.context,
    %input: tensor<1x3x16x16xf16>,
    %output: tensor<1x4x32x32xf16>) {
  // expected-error @+1 {{'hip.resize' op resize output dimension 1 must match input N/C extent 3}}
  %result = hip.resize(%ctx)
      ins(%input : tensor<1x3x16x16xf16>)
      outs(%output : tensor<1x4x32x32xf16>)
      {mode = 1, coord_transform = 2, nearest_mode = 3}
      : tensor<1x4x32x32xf16>
  return
}
