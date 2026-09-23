// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.top_k lowers inside a rock.kernel function. TOSA has no sort and
// no top-k, so this expands into K rounds of reduce_max + argmax, masking the
// position just taken so the next round finds the runner-up. The mask compares
// an iota against the round's argmax rather than comparing values, so a tie
// loses only one of its members per round.
//
// The masked-in value has to be one the input cannot hold, or a position
// would stay tied with its own mask and be picked twice. Floats use NaN,
// which the IGNORE reduction drops outright; integers run one width up and
// mask below the range they arrived in.
//
// Forms the expansion cannot express -- a K too wide to unroll, smallest-first
// on integers, i64, f64 -- stay hip ops rather than failing the pass.
//
// FILE LAYOUT:
// Converting cases first, then the cases left alone. Each rejection that must
// fail the pass gets its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// K = 2 runs two rounds. Only the first masks, since the last round has
// nothing left to search, and the two rounds concat into the results.
// CHECK-LABEL: func.func @top_k_two
// CHECK: tosa.reduce_max
// CHECK: tosa.argmax
// CHECK: tosa.equal
// CHECK: tosa.select
// CHECK: tosa.reduce_max
// CHECK: tosa.argmax
// CHECK: tosa.concat
// CHECK: tosa.concat
// CHECK-NOT: hip.top_k
func.func @top_k_two(
    %ctx: !hip.context, %x: tensor<3x4xf32>, %k: tensor<i64>,
    %vinit: tensor<3x2xf32>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xf32>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xf32>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<3x2xf32>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xf32>, tensor<3x2xi64>
}

// K = 1 is a single round: no masking and no concat, just the reduction and
// its index. This is the common classification shape.
// CHECK-LABEL: func.func @top_k_one
// CHECK: tosa.reduce_max
// CHECK: tosa.argmax
// CHECK-NOT: tosa.select
// CHECK-NOT: tosa.concat
// CHECK-NOT: hip.top_k
func.func @top_k_one(
    %ctx: !hip.context, %x: tensor<3x4xf32>, %k: tensor<i64>,
    %vinit: tensor<1x4xf32>, %iinit: tensor<1x4xi64>)
    -> (tensor<1x4xf32>, tensor<1x4xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<1x4xf32>, tensor<1x4xi64>)
      {axis = 0 : i64, largest = true, sorted = true}
      : tensor<1x4xf32>, tensor<1x4xi64>
  return %values, %indices : tensor<1x4xf32>, tensor<1x4xi64>
}

// A negative axis counts from the back, and i32 indices need no widening cast.
// CHECK-LABEL: func.func @top_k_negative_axis
// CHECK: tosa.argmax
// CHECK-NOT: hip.top_k
func.func @top_k_negative_axis(
    %ctx: !hip.context, %x: tensor<2x3x5xf16>, %k: tensor<i64>,
    %vinit: tensor<2x3x2xf16>, %iinit: tensor<2x3x2xi32>)
    -> (tensor<2x3x2xf16>, tensor<2x3x2xi32>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<2x3x5xf16>, tensor<i64>)
      outs(%vinit, %iinit : tensor<2x3x2xf16>, tensor<2x3x2xi32>)
      {axis = -1 : i64, largest = true, sorted = true}
      : tensor<2x3x2xf16>, tensor<2x3x2xi32>
  return %values, %indices : tensor<2x3x2xf16>, tensor<2x3x2xi32>
}

// Smallest-first negates the input, runs the same largest-first rounds, and
// negates the values back. The indices need no such correction.
// CHECK-LABEL: func.func @top_k_smallest
// CHECK: tosa.negate
// CHECK: tosa.argmax
// CHECK: tosa.negate
// CHECK-NOT: hip.top_k
func.func @top_k_smallest(
    %ctx: !hip.context, %x: tensor<3x4xf32>, %k: tensor<i64>,
    %vinit: tensor<3x2xf32>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xf32>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xf32>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = false, sorted = true}
      : tensor<3x2xf32>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xf32>, tensor<3x2xi64>
}

// Integers take the largest-first form. The rounds run in i64 so the mask can
// sit one below the i32 range, and the selected values -- all of which the i32
// input held -- narrow back at the end.
// CHECK-LABEL: func.func @top_k_integer
// CHECK: tosa.cast %{{.*}} : (tensor<3x4xi32>) -> tensor<3x4xi64>
// CHECK: tosa.argmax
// CHECK: tosa.cast %{{.*}} : (tensor<3x2xi64>) -> tensor<3x2xi32>
// CHECK-NOT: hip.top_k
func.func @top_k_integer(
    %ctx: !hip.context, %x: tensor<3x4xi32>, %k: tensor<i64>,
    %vinit: tensor<3x2xi32>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xi32>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xi32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xi32>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<3x2xi32>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xi32>, tensor<3x2xi64>
}

// A K wider than the unroll bound would emit an unusable number of rounds, so
// it is left alone.
// CHECK-LABEL: func.func @top_k_too_wide
// CHECK: hip.top_k
// CHECK-NOT: tosa.argmax
func.func @top_k_too_wide(
    %ctx: !hip.context, %x: tensor<3x64xf32>, %k: tensor<i64>,
    %vinit: tensor<3x20xf32>, %iinit: tensor<3x20xi64>)
    -> (tensor<3x20xf32>, tensor<3x20xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x64xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x20xf32>, tensor<3x20xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<3x20xf32>, tensor<3x20xi64>
  return %values, %indices : tensor<3x20xf32>, tensor<3x20xi64>
}

// Smallest-first on integers would need to negate the signed minimum, so it is
// left alone rather than lowered wrongly.
// CHECK-LABEL: func.func @top_k_integer_smallest
// CHECK: hip.top_k
// CHECK-NOT: tosa.argmax
func.func @top_k_integer_smallest(
    %ctx: !hip.context, %x: tensor<3x4xi32>, %k: tensor<i64>,
    %vinit: tensor<3x2xi32>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xi32>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xi32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xi32>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = false, sorted = true}
      : tensor<3x2xi32>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xi32>, tensor<3x2xi64>
}

// The mask has to be below everything the input can hold, so the rounds run
// one integer width up. i64 has nowhere to go, so it is left alone rather
// than masked with a value the input could have held.
// CHECK-LABEL: func.func @top_k_integer_i64
// CHECK: hip.top_k
// CHECK-NOT: tosa.argmax
func.func @top_k_integer_i64(
    %ctx: !hip.context, %x: tensor<3x4xi64>, %k: tensor<i64>,
    %vinit: tensor<3x2xi64>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xi64>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xi64>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xi64>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<3x2xi64>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xi64>, tensor<3x2xi64>
}

// TOSA has no f64 tensor type.
// CHECK-LABEL: func.func @top_k_f64
// CHECK: hip.top_k
// CHECK-NOT: tosa.argmax
func.func @top_k_f64(
    %ctx: !hip.context, %x: tensor<3x4xf64>, %k: tensor<i64>,
    %vinit: tensor<3x2xf64>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xf64>, tensor<3x2xi64>) attributes {rock.kernel} {
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xf64>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xf64>, tensor<3x2xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<3x2xf64>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xf64>, tensor<3x2xi64>
}

// -----

// The results must agree with the input away from the selected axis.
func.func @top_k_off_axis_mismatch(
    %ctx: !hip.context, %x: tensor<3x4xf32>, %k: tensor<i64>,
    %vinit: tensor<5x2xf32>, %iinit: tensor<5x2xi64>)
    -> (tensor<5x2xf32>, tensor<5x2xi64>) attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.top_k'}}
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<5x2xf32>, tensor<5x2xi64>)
      {axis = 1 : i64, largest = true, sorted = true}
      : tensor<5x2xf32>, tensor<5x2xi64>
  return %values, %indices : tensor<5x2xf32>, tensor<5x2xi64>
}

// -----

// A result of the wrong rank is a shape disagreement, so it fails the pass
// rather than staying a hip op. The legality gate has to reach that failure
// without indexing the result by an axis the result does not have.
func.func @top_k_result_rank_mismatch(
    %ctx: !hip.context, %x: tensor<3x4x5xf32>, %k: tensor<i64>,
    %vinit: tensor<3x2xf32>, %iinit: tensor<3x2xi64>)
    -> (tensor<3x2xf32>, tensor<3x2xi64>) attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.top_k'}}
  %values, %indices = hip.top_k(%ctx)
      ins(%x, %k : tensor<3x4x5xf32>, tensor<i64>)
      outs(%vinit, %iinit : tensor<3x2xf32>, tensor<3x2xi64>)
      {axis = 2 : i64, largest = true, sorted = true}
      : tensor<3x2xf32>, tensor<3x2xi64>
  return %values, %indices : tensor<3x2xf32>, tensor<3x2xi64>
}
