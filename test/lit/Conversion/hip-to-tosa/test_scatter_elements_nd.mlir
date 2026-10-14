// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.scatter_elements and hip.scatter_nd lower inside a rock.kernel
// function. Both are the inverses of the matching gathers and reach
// tosa.scatter the same way: ScatterElements moves `axis` last and scatters
// with C = 1, ScatterND folds each index tuple into one linear index.
//
// Only reduction "none" is expressible -- tosa.scatter forbids repeating an
// output index, which is what the accumulating modes exist to define -- so
// those stay hip ops rather than failing the pass.
//
// FILE LAYOUT:
// Converting cases in the first chunk; each rejection in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Scattering the last axis needs no transpose: the leading dimensions already
// collapse into N.
// CHECK-LABEL: func.func @scatter_elements_last_axis
// CHECK-NOT: tosa.transpose
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK: tosa.scatter
// CHECK-NOT: hip.scatter_elements
func.func @scatter_elements_last_axis(
    %ctx: !hip.context, %data: tensor<2x4xf32>, %indices: tensor<2x3xi32>,
    %updates: tensor<2x3xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x4xf32>, tensor<2x3xi32>, tensor<2x3xf32>)
      outs(%init : tensor<2x4xf32>) {axis = 1 : i64, reduction = "none"}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// An inner axis moves to the back first. Data, indices and updates are each
// transposed in, and the scattered result is transposed back out.
// CHECK-LABEL: func.func @scatter_elements_inner_axis
// CHECK: tosa.transpose
// CHECK: tosa.transpose
// CHECK: tosa.transpose
// CHECK: tosa.scatter
// CHECK: tosa.transpose
// CHECK-NOT: hip.scatter_elements
func.func @scatter_elements_inner_axis(
    %ctx: !hip.context, %data: tensor<4x3xf32>, %indices: tensor<2x3xi32>,
    %updates: tensor<2x3xf32>, %init: tensor<4x3xf32>)
    -> tensor<4x3xf32> attributes {rock.kernel} {
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<4x3xf32>, tensor<2x3xi32>, tensor<2x3xf32>)
      outs(%init : tensor<4x3xf32>) {axis = 0 : i64, reduction = "none"}
      : tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

// A negative axis counts from the back, and i64 indices narrow to the i32
// tosa.scatter requires.
// CHECK-LABEL: func.func @scatter_elements_negative_axis
// CHECK: tosa.cast
// CHECK: tosa.scatter
// CHECK-NOT: hip.scatter_elements
func.func @scatter_elements_negative_axis(
    %ctx: !hip.context, %data: tensor<2x3x4xf16>, %indices: tensor<2x3x2xi64>,
    %updates: tensor<2x3x2xf16>, %init: tensor<2x3x4xf16>)
    -> tensor<2x3x4xf16> attributes {rock.kernel} {
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x3x4xf16>, tensor<2x3x2xi64>, tensor<2x3x2xf16>)
      outs(%init : tensor<2x3x4xf16>) {axis = -1 : i64, reduction = "none"}
      : tensor<2x3x4xf16>
  return %result : tensor<2x3x4xf16>
}

// A one-wide tuple is already the linear index into the single scattered
// dimension, so it needs no stride arithmetic.
// CHECK-LABEL: func.func @scatter_nd_single_index
// CHECK-NOT: tosa.transpose
// CHECK: tosa.slice
// CHECK: tosa.scatter
// CHECK-NOT: tosa.mul
// CHECK-NOT: hip.scatter_nd
func.func @scatter_nd_single_index(
    %ctx: !hip.context, %data: tensor<4x3xf32>, %indices: tensor<2x1xi64>,
    %updates: tensor<2x3xf32>, %init: tensor<4x3xf32>)
    -> tensor<4x3xf32> attributes {rock.kernel} {
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<4x3xf32>, tensor<2x1xi64>, tensor<2x3xf32>)
      outs(%init : tensor<4x3xf32>) {reduction = "none"}
      : tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

// A two-wide tuple indexes data[0] and data[1], which flatten into one K of
// 2*3 = 6. The leading component carries stride 3 and the trailing one stride
// 1, so one multiply and one add fold the pair into a linear index.
// CHECK-LABEL: func.func @scatter_nd_index_tuple
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK: tosa.scatter
// CHECK-NOT: hip.scatter_nd
func.func @scatter_nd_index_tuple(
    %ctx: !hip.context, %data: tensor<2x3x4xf32>, %indices: tensor<5x2xi32>,
    %updates: tensor<5x4xf32>, %init: tensor<2x3x4xf32>)
    -> tensor<2x3x4xf32> attributes {rock.kernel} {
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x3x4xf32>, tensor<5x2xi32>, tensor<5x4xf32>)
      outs(%init : tensor<2x3x4xf32>) {reduction = "none"}
      : tensor<2x3x4xf32>
  return %result : tensor<2x3x4xf32>
}

// When the tuple spans data's whole rank each update is a scalar, so C is 1.
// CHECK-LABEL: func.func @scatter_nd_full_tuple
// CHECK: tosa.scatter
// CHECK-NOT: hip.scatter_nd
func.func @scatter_nd_full_tuple(
    %ctx: !hip.context, %data: tensor<2x3xf32>, %indices: tensor<4x2xi32>,
    %updates: tensor<4xf32>, %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> attributes {rock.kernel} {
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x3xf32>, tensor<4x2xi32>, tensor<4xf32>)
      outs(%init : tensor<2x3xf32>) {reduction = "none"}
      : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// The accumulating modes are left alone rather than failing the pass: they
// need a read-modify-write and permit the repeated index tosa.scatter bans.
// CHECK-LABEL: func.func @scatter_elements_reduction_add
// CHECK: hip.scatter_elements
// CHECK-NOT: tosa.scatter
func.func @scatter_elements_reduction_add(
    %ctx: !hip.context, %data: tensor<2x4xf32>, %indices: tensor<2x3xi32>,
    %updates: tensor<2x3xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x4xf32>, tensor<2x3xi32>, tensor<2x3xf32>)
      outs(%init : tensor<2x4xf32>) {axis = 1 : i64, reduction = "add"}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @scatter_nd_reduction_mul
// CHECK: hip.scatter_nd
// CHECK-NOT: tosa.scatter
func.func @scatter_nd_reduction_mul(
    %ctx: !hip.context, %data: tensor<4x3xf32>, %indices: tensor<2x1xi32>,
    %updates: tensor<2x3xf32>, %init: tensor<4x3xf32>)
    -> tensor<4x3xf32> attributes {rock.kernel} {
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<4x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>)
      outs(%init : tensor<4x3xf32>) {reduction = "mul"}
      : tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

// -----

// More updates than the scattered axis holds must repeat an output index,
// which tosa.scatter bans and ONNX does not define under reduction "none".
func.func @scatter_elements_more_updates_than_axis(
    %ctx: !hip.context, %data: tensor<2x2xf32>, %indices: tensor<2x5xi32>,
    %updates: tensor<2x5xf32>, %init: tensor<2x2xf32>)
    -> tensor<2x2xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.scatter_elements'}}
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x2xf32>, tensor<2x5xi32>, tensor<2x5xf32>)
      outs(%init : tensor<2x2xf32>) {axis = 1 : i64, reduction = "none"}
      : tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}

// -----

// Off the scattered axis an update lands at its own coordinate, so a
// disagreeing extent has no meaning.
func.func @scatter_elements_off_axis_mismatch(
    %ctx: !hip.context, %data: tensor<2x4xf32>, %indices: tensor<3x3xi32>,
    %updates: tensor<3x3xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.scatter_elements'}}
  %result = hip.scatter_elements(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x4xf32>, tensor<3x3xi32>, tensor<3x3xf32>)
      outs(%init : tensor<2x4xf32>) {axis = 1 : i64, reduction = "none"}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// -----

// Same for ScatterND: W may not exceed the flattened scattered range K.
func.func @scatter_nd_more_updates_than_range(
    %ctx: !hip.context, %data: tensor<2x3xf32>, %indices: tensor<5x1xi32>,
    %updates: tensor<5x3xf32>, %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.scatter_nd'}}
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x3xf32>, tensor<5x1xi32>, tensor<5x3xf32>)
      outs(%init : tensor<2x3xf32>) {reduction = "none"}
      : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// -----

// The tuple names more dimensions than data has, so it indexes nothing.
func.func @scatter_nd_tuple_too_wide(
    %ctx: !hip.context, %data: tensor<2x3xf32>, %indices: tensor<4x3xi32>,
    %updates: tensor<4xf32>, %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.scatter_nd'}}
  %result = hip.scatter_nd(%ctx)
      ins(%data, %indices, %updates :
          tensor<2x3xf32>, tensor<4x3xi32>, tensor<4xf32>)
      outs(%init : tensor<2x3xf32>) {reduction = "none"}
      : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
