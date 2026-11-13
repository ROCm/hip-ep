// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.grid_sample expands to TOSA gather + interpolate inside a
// rock.kernel function. TOSA has no grid_sample; tosa.resize only scales a
// regular lattice. The expansion matches hip_grid_sample: unnormalize the
// (x, y) grid, gather the spatial plane, bilinear-lerp or nearest, with
// zeros or border padding.
//
// FILE LAYOUT:
// Converting cases live in the first --split-input-file chunk. Each rejected
// form gets its own chunk so a legalization failure cannot mask later cases.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @grid_sample_bilinear
// CHECK: tosa.gather
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK-NOT: hip.grid_sample
func.func @grid_sample_bilinear(
    %ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
    %grid: tensor<1x4x4x2xf32>, %init: tensor<1x3x4x4xf32>)
    -> tensor<1x3x4x4xf32> attributes {rock.kernel} {
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>)
           outs(%init : tensor<1x3x4x4xf32>)
           {mode = 1, padding_mode = 0, align_corners = 0}
           : tensor<1x3x4x4xf32>
  return %r : tensor<1x3x4x4xf32>
}

// CHECK-LABEL: func.func @grid_sample_nearest_border
// CHECK: tosa.gather
// CHECK-NOT: hip.grid_sample
func.func @grid_sample_nearest_border(
    %ctx: !hip.context, %x: tensor<1x2x4x4xf16>,
    %grid: tensor<1x2x2x2xf16>, %init: tensor<1x2x2x2xf16>)
    -> tensor<1x2x2x2xf16> attributes {rock.kernel} {
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x2x4x4xf16>, tensor<1x2x2x2xf16>)
           outs(%init : tensor<1x2x2x2xf16>)
           {align_corners = 1, mode = 0, padding_mode = 1}
           : tensor<1x2x2x2xf16>
  return %r : tensor<1x2x2x2xf16>
}

// CHECK-LABEL: func.func @grid_sample_outlined_kernel
// CHECK: tosa.gather
// CHECK-NOT: hip.grid_sample
func.func @grid_sample_outlined_kernel(
    %x: tensor<1x3x8x8xf32>, %grid: tensor<1x4x4x2xf32>,
    %init: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf32>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>)
           outs(%init : tensor<1x3x4x4xf32>)
           {mode = 1, padding_mode = 0, align_corners = 0}
           : tensor<1x3x4x4xf32>
  return %r : tensor<1x3x4x4xf32>
}

// -----

func.func @grid_sample_dynamic(
    %ctx: !hip.context, %x: tensor<1x3x?x?xf32>,
    %grid: tensor<1x?x?x2xf32>, %init: tensor<1x3x?x?xf32>)
    -> tensor<1x3x?x?xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.grid_sample'}}
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x3x?x?xf32>, tensor<1x?x?x2xf32>)
           outs(%init : tensor<1x3x?x?xf32>)
           {mode = 1, padding_mode = 0, align_corners = 0}
           : tensor<1x3x?x?xf32>
  return %r : tensor<1x3x?x?xf32>
}

// -----

func.func @grid_sample_integer(
    %ctx: !hip.context, %x: tensor<1x3x8x8xi32>,
    %grid: tensor<1x4x4x2xi32>, %init: tensor<1x3x4x4xi32>)
    -> tensor<1x3x4x4xi32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.grid_sample'}}
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x3x8x8xi32>, tensor<1x4x4x2xi32>)
           outs(%init : tensor<1x3x4x4xi32>)
           {mode = 1, padding_mode = 0, align_corners = 0}
           : tensor<1x3x4x4xi32>
  return %r : tensor<1x3x4x4xi32>
}

// -----

func.func @grid_sample_reflection(
    %ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
    %grid: tensor<1x4x4x2xf32>, %init: tensor<1x3x4x4xf32>)
    -> tensor<1x3x4x4xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.grid_sample'}}
  %r = hip.grid_sample(%ctx)
           ins(%x, %grid : tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>)
           outs(%init : tensor<1x3x4x4xf32>)
           {mode = 1, padding_mode = 2, align_corners = 0}
           : tensor<1x3x4x4xf32>
  return %r : tensor<1x3x4x4xf32>
}
