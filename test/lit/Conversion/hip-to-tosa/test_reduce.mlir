// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.reduce_sum / hip.reduce_mean lower inside a rock.kernel
// function. TOSA reduce ops take one i32 axis and always keepdims=1;
// keepdims=0 is a reshape afterward. Mean is scale-by-1/N then reduce_sum.
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

// CHECK-LABEL: func.func @reduce_sum_not_a_kernel
// CHECK: hip.reduce_sum
// CHECK-NOT: tosa.reduce_sum
func.func @reduce_sum_not_a_kernel(%ctx: !hip.context, %data: tensor<2x8xf16>,
                                   %init: tensor<2x1xf16>) -> tensor<2x1xf16> {
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.reduce_sum(%ctx)
         ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
         outs(%init : tensor<2x1xf16>) : tensor<2x1xf16>
  return %r : tensor<2x1xf16>
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
