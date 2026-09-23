// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.gather_elements and hip.gather_nd lower inside a rock.kernel
// function. Both reach TOSA's one batched gather, [N,K,C] x [N,W] -> [N,W,C],
// but from opposite directions: GatherElements reads a single element per
// output position, so it gathers with C = 1 and moves `axis` last to collapse
// the rest into N; GatherND reads a slice per index tuple, so it folds the
// tuple into one linear index and lets `batch_dims` be N directly.
//
// FILE LAYOUT:
// Converting cases in the first chunk; each rejection in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// Gathering the last axis needs no transpose: the leading dimensions are
// already contiguous, so they collapse straight into N.
// CHECK-LABEL: func.func @gather_elements_last_axis
// CHECK-NOT: tosa.transpose
// CHECK: tosa.greater
// CHECK: tosa.add
// CHECK: tosa.select
// CHECK: tosa.gather
// CHECK-NOT: hip.gather_elements
func.func @gather_elements_last_axis(
    %ctx: !hip.context, %data: tensor<2x3xf32>,
    %indices: tensor<2x4xi32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %result = hip.gather_elements(%ctx)
      ins(%data, %indices : tensor<2x3xf32>, tensor<2x4xi32>)
      outs(%init : tensor<2x4xf32>) {axis = 1 : i64}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// An inner axis moves to the back before the gather and returns afterward, so
// data and indices are each transposed in and the result is transposed out.
// CHECK-LABEL: func.func @gather_elements_inner_axis
// CHECK: tosa.transpose
// CHECK: tosa.transpose
// CHECK: tosa.gather
// CHECK: tosa.transpose
// CHECK-NOT: hip.gather_elements
func.func @gather_elements_inner_axis(
    %ctx: !hip.context, %data: tensor<3x4xf32>,
    %indices: tensor<2x4xi32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  %result = hip.gather_elements(%ctx)
      ins(%data, %indices : tensor<3x4xf32>, tensor<2x4xi32>)
      outs(%init : tensor<2x4xf32>) {axis = 0 : i64}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// A negative axis counts from the back, and i64 indices narrow to the i32
// tosa.gather requires.
// CHECK-LABEL: func.func @gather_elements_negative_axis
// CHECK: tosa.cast
// CHECK: tosa.gather
// CHECK-NOT: hip.gather_elements
func.func @gather_elements_negative_axis(
    %ctx: !hip.context, %data: tensor<2x3x4xf16>,
    %indices: tensor<2x3x5xi64>, %init: tensor<2x3x5xf16>)
    -> tensor<2x3x5xf16> attributes {rock.kernel} {
  %result = hip.gather_elements(%ctx)
      ins(%data, %indices : tensor<2x3x4xf16>, tensor<2x3x5xi64>)
      outs(%init : tensor<2x3x5xf16>) {axis = -1 : i64}
      : tensor<2x3x5xf16>
  return %result : tensor<2x3x5xf16>
}

// A one-wide index tuple is already the linear index into the single gathered
// dimension, so it is normalized and used as-is with no stride arithmetic.
// CHECK-LABEL: func.func @gather_nd_single_index
// CHECK-NOT: tosa.transpose
// CHECK: tosa.slice
// CHECK: tosa.gather
// CHECK-NOT: tosa.mul
// CHECK-NOT: hip.gather_nd
func.func @gather_nd_single_index(
    %ctx: !hip.context, %data: tensor<4x3xf32>,
    %indices: tensor<2x1xi64>, %init: tensor<2x3xf32>)
    -> tensor<2x3xf32> attributes {rock.kernel} {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<4x3xf32>, tensor<2x1xi64>)
      outs(%init : tensor<2x3xf32>) {batch_dims = 0 : i64}
      : tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

// A two-wide tuple indexes data[0] and data[1], which flatten into one K of
// 2*3 = 6. The leading component carries stride 3, the trailing one stride 1,
// so exactly one multiply and one add fold the pair into a linear index.
// CHECK-LABEL: func.func @gather_nd_index_tuple
// CHECK: tosa.mul
// CHECK: tosa.add
// CHECK: tosa.gather
// CHECK-NOT: hip.gather_nd
func.func @gather_nd_index_tuple(
    %ctx: !hip.context, %data: tensor<2x3x4xf32>,
    %indices: tensor<5x2xi32>, %init: tensor<5x4xf32>)
    -> tensor<5x4xf32> attributes {rock.kernel} {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<2x3x4xf32>, tensor<5x2xi32>)
      outs(%init : tensor<5x4xf32>) {batch_dims = 0 : i64}
      : tensor<5x4xf32>
  return %result : tensor<5x4xf32>
}

// Batch dims map onto TOSA's N unchanged, so a batched GatherND costs no
// transpose and no extra arithmetic over the unbatched one.
// CHECK-LABEL: func.func @gather_nd_batch_dims
// CHECK-NOT: tosa.transpose
// CHECK: tosa.gather
// CHECK-NOT: hip.gather_nd
func.func @gather_nd_batch_dims(
    %ctx: !hip.context, %data: tensor<2x3x4xf32>,
    %indices: tensor<2x5x1xi32>, %init: tensor<2x5x4xf32>)
    -> tensor<2x5x4xf32> attributes {rock.kernel} {
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<2x3x4xf32>, tensor<2x5x1xi32>)
      outs(%init : tensor<2x5x4xf32>) {batch_dims = 1 : i64}
      : tensor<2x5x4xf32>
  return %result : tensor<2x5x4xf32>
}

// -----

// Off the gathered axis each output position reads data at its own
// coordinate, so a disagreeing extent has no meaning.
func.func @gather_elements_off_axis_mismatch(
    %ctx: !hip.context, %data: tensor<2x3xf32>,
    %indices: tensor<4x3xi32>, %init: tensor<4x3xf32>)
    -> tensor<4x3xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gather_elements'}}
  %result = hip.gather_elements(%ctx)
      ins(%data, %indices : tensor<2x3xf32>, tensor<4x3xi32>)
      outs(%init : tensor<4x3xf32>) {axis = 1 : i64}
      : tensor<4x3xf32>
  return %result : tensor<4x3xf32>
}

// -----

// Floating-point indices are not ONNX GatherElements and would leave a
// tosa.gather with a non-i32 index operand.
func.func @gather_elements_float_indices(
    %ctx: !hip.context, %data: tensor<2x3xf32>,
    %indices: tensor<2x4xf32>, %init: tensor<2x4xf32>)
    -> tensor<2x4xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gather_elements'}}
  %result = hip.gather_elements(%ctx)
      ins(%data, %indices : tensor<2x3xf32>, tensor<2x4xf32>)
      outs(%init : tensor<2x4xf32>) {axis = 1 : i64}
      : tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// -----

// The tuple names more dimensions than data has, so it indexes nothing.
func.func @gather_nd_tuple_too_wide(
    %ctx: !hip.context, %data: tensor<2x3xf32>,
    %indices: tensor<4x3xi32>, %init: tensor<4xf32>)
    -> tensor<4xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gather_nd'}}
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<2x3xf32>, tensor<4x3xi32>)
      outs(%init : tensor<4xf32>) {batch_dims = 0 : i64}
      : tensor<4xf32>
  return %result : tensor<4xf32>
}

// -----

// Batch dims have to be a shared prefix; a disagreement means the two do not
// describe the same batch.
func.func @gather_nd_batch_dims_mismatch(
    %ctx: !hip.context, %data: tensor<2x3x4xf32>,
    %indices: tensor<5x6x1xi32>, %init: tensor<5x6x4xf32>)
    -> tensor<5x6x4xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.gather_nd'}}
  %result = hip.gather_nd(%ctx)
      ins(%data, %indices : tensor<2x3x4xf32>, tensor<5x6x1xi32>)
      outs(%init : tensor<5x6x4xf32>) {batch_dims = 1 : i64}
      : tensor<5x6x4xf32>
  return %result : tensor<5x6x4xf32>
}
