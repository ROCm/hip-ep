// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify static hip.one_hot expands to supported TOSA elementwise ops.
//
// TOSA has no OneHot or NonZero op. OneHot has a static-shape decomposition:
// normalize negative indices, compare them with a tiled class range, and
// select the on/off values. NonZero cannot be represented by supported TOSA
// ops: its ordered compact output requires a prefix scan and its [rank, N]
// shape and count result depend on input values. It therefore remains a HIP
// runtime boundary rather than receiving an incorrect conversion.
//
// FILE LAYOUT:
// Converting and pass-through cases live in the first split-input-file chunk.
// Each rejected OneHot form gets its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Default axis appends the depth dimension.
// CHECK-LABEL: func.func @one_hot_default_axis
// CHECK: tosa.greater
// CHECK: tosa.add
// CHECK: tosa.select
// CHECK: tosa.tile
// CHECK: tosa.equal
// CHECK: tosa.select
// CHECK-NOT: hip.one_hot
func.func @one_hot_default_axis(
    %ctx: !hip.context, %indices: tensor<2xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %r = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2xi64>, tensor<i64>, tensor<2xf32>)
      outs(%init : tensor<2x4xf32>) : tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// An inner insertion axis exercises reshape/tile shape construction.
// CHECK-LABEL: func.func @one_hot_axis1
// CHECK: tosa.reshape
// CHECK: tosa.tile
// CHECK: tosa.equal
// CHECK: tosa.select
// CHECK-NOT: hip.one_hot
func.func @one_hot_axis1(
    %ctx: !hip.context, %indices: tensor<2x2xi32>, %depth: tensor<i64>,
    %values: tensor<2xf16>, %init: tensor<2x3x2xf16>)
    -> tensor<2x3x2xf16> attributes {rock.kernel} {
  %r = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2x2xi32>, tensor<i64>, tensor<2xf16>)
      outs(%init : tensor<2x3x2xf16>) {axis = 1 : i64}
      : tensor<2x3x2xf16>
  return %r : tensor<2x3x2xf16>
}

// The shape hip-fuse-rocmlir produces for an outlined kernel.
// CHECK-LABEL: func.func @one_hot_outlined_kernel
// CHECK: tosa.equal
// CHECK-NOT: hip.one_hot
func.func @one_hot_outlined_kernel(
    %indices: tensor<2xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2xi64>, tensor<i64>, tensor<2xf32>)
      outs(%init : tensor<2x4xf32>) : tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// NonZero deliberately stays in HIP. Its second result is the
// data-dependent count used by hip.readback_dim in the production pipeline.
// CHECK-LABEL: func.func @nonzero_runtime_boundary
// CHECK: hip.nonzero
// CHECK-NOT: tosa.
func.func @nonzero_runtime_boundary(
    %ctx: !hip.context, %x: tensor<3x4xf32>,
    %init: tensor<2x?xi64>, %countInit: tensor<i32>)
    -> tensor<2x?xi64> attributes {rock.kernel} {
  %y, %count = hip.nonzero(%ctx) ins(%x : tensor<3x4xf32>)
      outs(%init, %countInit : tensor<2x?xi64>, tensor<i32>)
      {input_data_type = 0 : i64} : tensor<2x?xi64>, tensor<i32>
  return %y : tensor<2x?xi64>
}

// -----

// Runtime depth gives the output a dynamic axis, which static TOSA tile and
// class constants cannot represent.
func.func @one_hot_dynamic_depth(
    %ctx: !hip.context, %indices: tensor<2xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %init: tensor<2x?xf32>)
    -> tensor<2x?xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.one_hot'}}
  %r = hip.one_hot(%ctx)
      ins(%indices, %depth, %values :
          tensor<2xi64>, tensor<i64>, tensor<2xf32>)
      outs(%init : tensor<2x?xf32>) : tensor<2x?xf32>
  return %r : tensor<2x?xf32>
}
