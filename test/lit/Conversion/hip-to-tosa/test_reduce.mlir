// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the hip reduction ops lower inside a rock.kernel function. TOSA
// reduce ops take one i32 axis and always keepdims=1; keepdims=0 is a reshape
// afterward.
//
// reduce_sum, reduce_max, reduce_min and reduce_prod share the ReduceConverter
// template, so the axis, keepdims and identity behaviour is exercised once
// (through hip.reduce_sum) rather than repeated per op; the per-op cases prove
// only the op mapping. reduce_mean and reduce_l2 have no TOSA counterpart and
// expand instead: mean is scale-by-1/N then reduce_sum, and l2 is
// sqrt(sum(x^2)) with the sqrt spelled reciprocal(rsqrt(..)).
//
// FILE LAYOUT:
// Converting cases in the first chunk; each rejection in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @reduce_sum
// CHECK: tosa.reduce_sum %arg1 {axis = 1 : i32}
// CHECK-NOT: hip.reduce_sum
func.func @reduce_sum(%ctx: !hip.context, %data: tensor<2x8xf16>,
                      %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @reduce_sum_keepdims0
// CHECK: tosa.reduce_sum %arg1 {axis = 1 : i32}
// CHECK: tosa.reshape
// CHECK-NOT: hip.reduce_sum
func.func @reduce_sum_keepdims0(%ctx: !hip.context, %data: tensor<2x8xf16>,
                                %init: tensor<2xf16>) -> tensor<2xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2xf16>)
         {keepdims = 0 : i64} : tensor<2xf16>
  return %r : tensor<2xf16>
}

// CHECK-LABEL: func.func @reduce_sum_neg_axis
// CHECK: tosa.reduce_sum %arg1 {axis = 1 : i32}
// CHECK-NOT: hip.reduce_sum
func.func @reduce_sum_neg_axis(%ctx: !hip.context, %data: tensor<2x8xf16>,
                               %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[-1]> : tensor<1xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @reduce_sum_i32
// CHECK: tosa.reduce_sum %arg1 {axis = 0 : i32}
// CHECK-NOT: hip.reduce_sum
func.func @reduce_sum_i32(%ctx: !hip.context, %data: tensor<4x8xi32>,
                          %init: tensor<1x8xi32>) -> tensor<1x8xi32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[0]> : tensor<1xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<4x8xi32>, tensor<1xi64>)
         outs(%init : tensor<1x8xi32>) : tensor<1x8xi32>
  return %r : tensor<1x8xi32>
}

// Empty axes + noop_with_empty_axes is an ONNX identity.
// CHECK-LABEL: func.func @reduce_sum_noop
// CHECK-NOT: tosa.reduce_sum
// CHECK-NOT: hip.reduce_sum
func.func @reduce_sum_noop(%ctx: !hip.context, %data: tensor<2x8xf16>,
                           %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[]> : tensor<0xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<0xi64>)
         outs(%init : tensor<2x8xf16>)
         {noop_with_empty_axes = 1 : i64} : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// The nan_mode attribute on tosa.reduce_max defaults to PROPAGATE, which is
// what ONNX ReduceMax does, so it is omitted from the pretty form.
// CHECK-LABEL: func.func @reduce_max
// CHECK: tosa.reduce_max %arg1 {axis = 1 : i32}
// CHECK-NOT: hip.reduce_max
func.func @reduce_max(%ctx: !hip.context, %data: tensor<2x8xf16>,
                      %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_max(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @reduce_min
// CHECK: tosa.reduce_min %arg1 {axis = 1 : i32}
// CHECK-NOT: hip.reduce_min
func.func @reduce_min(%ctx: !hip.context, %data: tensor<2x8xf16>,
                      %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_min(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @reduce_prod
// CHECK: tosa.reduce_product %arg1 {axis = 1 : i32}
// CHECK-NOT: hip.reduce_prod
func.func @reduce_prod(%ctx: !hip.context, %data: tensor<2x8xf16>,
                       %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_prod(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// Unlike max and min, these three go through the same template, so keepdims=0
// is checked once here rather than for each of them.
// CHECK-LABEL: func.func @reduce_max_keepdims0
// CHECK: tosa.reduce_max %arg1 {axis = 1 : i32}
// CHECK: tosa.reshape
// CHECK-NOT: hip.reduce_max
func.func @reduce_max_keepdims0(%ctx: !hip.context, %data: tensor<2x8xf16>,
                                %init: tensor<2xf16>) -> tensor<2xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_max(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2xf16>)
         {keepdims = 0 : i64} : tensor<2xf16>
  return %r : tensor<2xf16>
}

// Integers reduce through the same op; only max/min/prod accept them, since
// the mean and l2 expansions are float-only.
// CHECK-LABEL: func.func @reduce_prod_i32
// CHECK: tosa.reduce_product %arg1 {axis = 0 : i32}
func.func @reduce_prod_i32(%ctx: !hip.context, %data: tensor<4x8xi32>,
                           %init: tensor<1x8xi32>) -> tensor<1x8xi32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[0]> : tensor<1xi64>
  %r = hip.reduce_prod(%ctx)
         ins(%data, %axes : tensor<4x8xi32>, tensor<1xi64>)
         outs(%init : tensor<1x8xi32>) : tensor<1x8xi32>
  return %r : tensor<1x8xi32>
}

// sqrt(sum(x^2)): square with a self-multiply, reduce, then take the square
// root as reciprocal(rsqrt(..)) since TOSA has no sqrt.
//
// An f16 input is widened to f32 first, so the square and the accumulation
// cannot overflow: 300 is an ordinary f16 but 300^2 is 90000 against an f16
// maximum of 65504, and squaring in place would return +inf for the norm of a
// single 300. reduce_l2_f16_kernel widens the same way.
// CHECK-LABEL: func.func @reduce_l2
// CHECK: %[[W:.*]] = tosa.cast %arg1 : (tensor<2x8xf16>) -> tensor<2x8xf32>
// CHECK: %[[SQ:.*]] = tosa.mul %[[W]], %[[W]]
// CHECK: %[[SUM:.*]] = tosa.reduce_sum %[[SQ]] {axis = 1 : i32}
// CHECK-SAME: -> tensor<2x1xf32>
// CHECK: %[[RS:.*]] = tosa.rsqrt %[[SUM]]
// CHECK: %[[N:.*]] = tosa.reciprocal %[[RS]]
// CHECK: tosa.cast %[[N]] : (tensor<2x1xf32>) -> tensor<2x1xf16>
// CHECK-NOT: hip.reduce_l2
func.func @reduce_l2(%ctx: !hip.context, %data: tensor<2x8xf16>,
                     %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// An f32 input is already the accumulation type, so it needs no cast on either
// side of the expansion.
// CHECK-LABEL: func.func @reduce_l2_f32
// CHECK-NOT: tosa.cast
// CHECK: %[[SQ:.*]] = tosa.mul %arg1, %arg1
// CHECK: tosa.reduce_sum %[[SQ]] {axis = 1 : i32}
// CHECK-NOT: tosa.cast
func.func @reduce_l2_f32(%ctx: !hip.context, %data: tensor<2x8xf32>,
                         %init: tensor<2x1xf32>) -> tensor<2x1xf32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xf32>, tensor<1xi64>)
         outs(%init : tensor<2x1xf32>) : tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

// The reshape for keepdims=0 comes after the whole expansion, including the
// narrowing cast, not after the reduce_sum in the middle of it.
// CHECK-LABEL: func.func @reduce_l2_keepdims0
// CHECK: tosa.reduce_sum {{.*}}{axis = 1 : i32}
// CHECK: tosa.rsqrt
// CHECK: tosa.reciprocal
// CHECK: %[[C:.*]] = tosa.cast {{.*}} -> tensor<2x1xf16>
// CHECK: tosa.reshape %[[C]]
// CHECK-NOT: hip.reduce_l2
func.func @reduce_l2_keepdims0(%ctx: !hip.context, %data: tensor<2x8xf16>,
                               %init: tensor<2xf16>) -> tensor<2xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2xf16>)
         {keepdims = 0 : i64} : tensor<2xf16>
  return %r : tensor<2xf16>
}

// noop_with_empty_axes reduces nothing, so ONNX defines the result as the
// input itself rather than as an elementwise norm.
// CHECK-LABEL: func.func @reduce_l2_noop
// CHECK-NOT: tosa.reduce_sum
// CHECK-NOT: tosa.rsqrt
// CHECK-NOT: hip.reduce_l2
func.func @reduce_l2_noop(%ctx: !hip.context, %data: tensor<2x8xf16>,
                          %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[]> : tensor<0xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<0xi64>)
         outs(%init : tensor<2x8xf16>)
         {noop_with_empty_axes = 1 : i64} : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @reduce_mean
// CHECK: tosa.reciprocal
// CHECK: tosa.reshape
// CHECK: tosa.mul
// CHECK: tosa.reduce_sum {{.*}}{axis = 1 : i32}
// CHECK-NOT: hip.reduce_mean
func.func @reduce_mean(%ctx: !hip.context, %data: tensor<2x8xf16>,
                       %init: tensor<2x1xf16>) -> tensor<2x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_mean(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @reduce_mean_outlined_kernel
// CHECK: tosa.reduce_sum {{.*}}{axis = 1 : i32}
// CHECK-NOT: hip.reduce_mean
func.func @reduce_mean_outlined_kernel(%data: tensor<2x8xf16>,
                                       %init: tensor<2x1xf16>)
    -> tensor<2x1xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_mean(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Forms this pass does not claim. reduce_max, reduce_min, reduce_prod and
// reduce_l2 are legal by element type *and* by structure, so an unsupported
// one is left as a hip op for the runtime lowering and the pass still succeeds
// rather than failing the whole function. These share one chunk since none
// produce a diagnostic.
//
// The structural half matters as much as the element type: the legality
// predicates run matchHipReduceCore, the same match the patterns run, so a
// dynamic, multi-axis or non-constant-axis reduction is declined up front
// instead of being claimed and then refused. reduce_sum and reduce_mean are
// deliberately not in this list -- they stay outright illegal, which is
// pre-existing behaviour covered by the diagnostics at the end of this file.
//===----------------------------------------------------------------------===//

// ONNX ReduceMax/ReduceMin accept unsigned types and OnnxToHip preserves them,
// but no TOSA lowering takes an unsigned tensor, so the whole unsigned family
// stays on the runtime path. For max and min there is a second reason not to
// reinterpret the bits as signless: a ui8 255 would read as -1 and lose a
// maximum it should win.
// CHECK-LABEL: func.func @reduce_max_unsigned
// CHECK: hip.reduce_max
// CHECK-NOT: tosa.reduce_max
func.func @reduce_max_unsigned(%ctx: !hip.context, %data: tensor<2x8xui8>,
                               %init: tensor<2x1xui8>) -> tensor<2x1xui8>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_max(%ctx)
         ins(%data, %axes : tensor<2x8xui8>, tensor<1xi64>)
         outs(%init : tensor<2x1xui8>) : tensor<2x1xui8>
  return %r : tensor<2x1xui8>
}

// CHECK-LABEL: func.func @reduce_min_unsigned
// CHECK: hip.reduce_min
// CHECK-NOT: tosa.reduce_min
func.func @reduce_min_unsigned(%ctx: !hip.context, %data: tensor<2x8xui32>,
                               %init: tensor<2x1xui32>) -> tensor<2x1xui32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_min(%ctx)
         ins(%data, %axes : tensor<2x8xui32>, tensor<1xi64>)
         outs(%init : tensor<2x1xui32>) : tensor<2x1xui32>
  return %r : tensor<2x1xui32>
}

// reduce_prod is gated on the element type too. It is sign-agnostic, so an
// unsigned product is fine, but f64 is not: Tosa_FloatTensor is AnyFloat, so
// tosa.reduce_product would verify and then have no lowering.
// CHECK-LABEL: func.func @reduce_prod_f64
// CHECK: hip.reduce_prod
// CHECK-NOT: tosa.reduce_product
func.func @reduce_prod_f64(%ctx: !hip.context, %data: tensor<2x8xf64>,
                           %init: tensor<2x1xf64>) -> tensor<2x1xf64>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_prod(%ctx)
         ins(%data, %axes : tensor<2x8xf64>, tensor<1xi64>)
         outs(%init : tensor<2x1xf64>) : tensor<2x1xf64>
  return %r : tensor<2x1xf64>
}

// Unsigned is declined for the sign-agnostic reductions too, and not because
// the arithmetic would differ -- two's-complement multiply gives the same bits
// whichever way the sign bit is read. It is that no TOSA lowering accepts an
// unsigned tensor at all: tosa.reduce_product on ui32 fails tosa-to-linalg
// with "'arith.constant' op integer return type must be signless", where the
// signless form becomes a linalg.reduce.
// CHECK-LABEL: func.func @reduce_prod_unsigned
// CHECK: hip.reduce_prod
// CHECK-NOT: tosa.reduce_product
func.func @reduce_prod_unsigned(%ctx: !hip.context, %data: tensor<2x8xui32>,
                                %init: tensor<2x1xui32>) -> tensor<2x1xui32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_prod(%ctx)
         ins(%data, %axes : tensor<2x8xui32>, tensor<1xi64>)
         outs(%init : tensor<2x1xui32>) : tensor<2x1xui32>
  return %r : tensor<2x1xui32>
}

// Structure, not just element type: an f16 reduce_max is a type this pass can
// express, but a multi-axis reduction is not a single tosa.reduce. Claiming it
// on the element type alone would abort the function's whole conversion.
// CHECK-LABEL: func.func @reduce_max_multi_axis
// CHECK: hip.reduce_max
// CHECK-NOT: tosa.reduce_max
func.func @reduce_max_multi_axis(%ctx: !hip.context, %data: tensor<2x8xf16>,
                                 %init: tensor<1x1xf16>) -> tensor<1x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[0, 1]> : tensor<2xi64>
  %r = hip.reduce_max(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<2xi64>)
         outs(%init : tensor<1x1xf16>) : tensor<1x1xf16>
  return %r : tensor<1x1xf16>
}

// Likewise a non-constant axes operand, which the pattern cannot read.
// CHECK-LABEL: func.func @reduce_min_nonconstant_axes
// CHECK: hip.reduce_min
// CHECK-NOT: tosa.reduce_min
func.func @reduce_min_nonconstant_axes(%ctx: !hip.context,
                                       %data: tensor<2x8xf32>,
                                       %axes: tensor<1xi64>,
                                       %init: tensor<2x1xf32>)
    -> tensor<2x1xf32> attributes {rock.kernel} {
  %r = hip.reduce_min(%ctx)
         ins(%data, %axes : tensor<2x8xf32>, tensor<1xi64>)
         outs(%init : tensor<2x1xf32>) : tensor<2x1xf32>
  return %r : tensor<2x1xf32>
}

// And a dynamic shape, which gives the pattern no static extent to reduce.
// CHECK-LABEL: func.func @reduce_prod_dynamic
// CHECK: hip.reduce_prod
// CHECK-NOT: tosa.reduce_product
func.func @reduce_prod_dynamic(%ctx: !hip.context, %data: tensor<?x8xf32>,
                               %init: tensor<?x1xf32>) -> tensor<?x1xf32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_prod(%ctx)
         ins(%data, %axes : tensor<?x8xf32>, tensor<1xi64>)
         outs(%init : tensor<?x1xf32>) : tensor<?x1xf32>
  return %r : tensor<?x1xf32>
}

// TOSA's float tensor constraint is AnyFloat, so an f64 norm would satisfy the
// verifier and then have no lowering. onnx.ReduceL2 permits f64 input, so this
// is a form the pass has to decline rather than one that cannot arrive.
// CHECK-LABEL: func.func @reduce_l2_f64
// CHECK: hip.reduce_l2
// CHECK-NOT: tosa.reduce_sum
func.func @reduce_l2_f64(%ctx: !hip.context, %data: tensor<2x8xf64>,
                         %init: tensor<2x1xf64>) -> tensor<2x1xf64>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xf64>, tensor<1xi64>)
         outs(%init : tensor<2x1xf64>) : tensor<2x1xf64>
  return %r : tensor<2x1xf64>
}

// The l2 expansion ends in tosa.rsqrt / tosa.reciprocal, which are float-only,
// so an integer reduction has no spelling even though tosa.reduce_sum alone
// would take one.
// CHECK-LABEL: func.func @integer_l2
// CHECK: hip.reduce_l2
// CHECK-NOT: tosa.reduce_sum
func.func @integer_l2(%ctx: !hip.context, %data: tensor<2x8xi32>,
                      %init: tensor<2x1xi32>) -> tensor<2x1xi32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_l2(%ctx)
         ins(%data, %axes : tensor<2x8xi32>, tensor<1xi64>)
         outs(%init : tensor<2x1xi32>) : tensor<2x1xi32>
  return %r : tensor<2x1xi32>
}

// -----

func.func @dynamic_shape(%ctx: !hip.context, %data: tensor<?x8xf16>,
                         %init: tensor<?x1xf16>) -> tensor<?x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'hip.reduce_sum'}}
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<?x8xf16>, tensor<1xi64>)
         outs(%init : tensor<?x1xf16>) : tensor<?x1xf16>
  return %r : tensor<?x1xf16>
}

// -----

func.func @non_constant_axes(%ctx: !hip.context, %data: tensor<2x8xf16>,
                             %axes: tensor<1xi64>, %init: tensor<2x1xf16>)
    -> tensor<2x1xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.reduce_sum'}}
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
}

// -----

func.func @multi_axis(%ctx: !hip.context, %data: tensor<2x8xf16>,
                      %init: tensor<1x1xf16>) -> tensor<1x1xf16>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[0, 1]> : tensor<2xi64>
  // expected-error @+1 {{failed to legalize operation 'hip.reduce_sum'}}
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<2xi64>)
         outs(%init : tensor<1x1xf16>) : tensor<1x1xf16>
  return %r : tensor<1x1xf16>
}

// -----

func.func @integer_mean(%ctx: !hip.context, %data: tensor<2x8xi32>,
                        %init: tensor<2x1xi32>) -> tensor<2x1xi32>
    attributes {rock.kernel} {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  // expected-error @+1 {{failed to legalize operation 'hip.reduce_mean'}}
  %r = hip.reduce_mean(%ctx)
         ins(%data, %axes : tensor<2x8xi32>, tensor<1xi64>)
         outs(%init : tensor<2x1xi32>) : tensor<2x1xi32>
  return %r : tensor<2x1xi32>
}
