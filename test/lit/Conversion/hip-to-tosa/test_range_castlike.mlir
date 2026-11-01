// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Range and CastLike convert to TOSA inside a rock.kernel.
//
// TOSA has no range/arange op. Range is affine in the index, so a static
// length becomes start + delta * iota, and constant bounds fold that to a
// single tosa.const.
//
// CastLike has no hip op of its own. simplify-onnx rewrites it to onnx.Cast
// (the type donor is never read), which convert-onnx-to-hip lowers to
// hip.cast, and hip.cast is 1-1 with tosa.cast. An identity CastLike is
// forwarded before it ever reaches a hip op.
//
// FILE LAYOUT:
// Converting cases live in the first --split-input-file chunk. Each rejected
// form gets its own chunk so a legalization failure cannot mask later cases.
//
// The general hip.cast to tosa.cast coverage -- including the float-to-int and
// unsigned endpoints that route through rocMLIR tosa.custom -- lives in
// test_unary.mlir with the rest of the unary group. This file covers only what
// is specific to CastLike.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Range.
//===----------------------------------------------------------------------===//

// Constant bounds fold to the sequence itself.
// CHECK-LABEL: func.func @range_i64_const
// CHECK: tosa.const{{.*}}dense<[2, 5, 8, 11]> : tensor<4xi64>
// CHECK-NOT: hip.range
func.func @range_i64_const(%ctx: !hip.context) -> tensor<4xi64>
    attributes {rock.kernel} {
  %start = arith.constant dense<2> : tensor<i64>
  %limit = arith.constant dense<14> : tensor<i64>
  %delta = arith.constant dense<3> : tensor<i64>
  %init = tensor.empty() : tensor<4xi64>
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i64>, tensor<i64>, tensor<i64>)
                       outs(%init : tensor<4xi64>) : tensor<4xi64>
  return %r : tensor<4xi64>
}

// A negative delta counts down.
// CHECK-LABEL: func.func @range_i32_negative_delta
// CHECK: tosa.const{{.*}}dense<[10, 8, 6]> : tensor<3xi32>
func.func @range_i32_negative_delta(%ctx: !hip.context) -> tensor<3xi32>
    attributes {rock.kernel} {
  %start = arith.constant dense<10> : tensor<i32>
  %limit = arith.constant dense<4> : tensor<i32>
  %delta = arith.constant dense<-2> : tensor<i32>
  %init = tensor.empty() : tensor<3xi32>
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i32>, tensor<i32>, tensor<i32>)
                       outs(%init : tensor<3xi32>) : tensor<3xi32>
  return %r : tensor<3xi32>
}

// Float bounds fold in the result element type.
// CHECK-LABEL: func.func @range_f32_const
// CHECK: tosa.const{{.*}}tensor<3xf32>
// CHECK-NOT: hip.range
func.func @range_f32_const(%ctx: !hip.context) -> tensor<3xf32>
    attributes {rock.kernel} {
  %start = arith.constant dense<1.000000e+00> : tensor<f32>
  %limit = arith.constant dense<2.500000e+00> : tensor<f32>
  %delta = arith.constant dense<5.000000e-01> : tensor<f32>
  %init = tensor.empty() : tensor<3xf32>
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<f32>, tensor<f32>, tensor<f32>)
                       outs(%init : tensor<3xf32>) : tensor<3xf32>
  return %r : tensor<3xf32>
}

// Runtime bounds with a static length scale the iota. The limit is unused:
// the trip count it determines is already the result extent.
// CHECK-LABEL: func.func @range_dynamic_bounds
// CHECK: %[[START:.*]] = tosa.reshape %arg1
// CHECK: %[[DELTA:.*]] = tosa.reshape %arg3
// CHECK: %[[IOTA:.*]] = "tosa.const"() <{values = dense<[0, 1, 2, 3]> : tensor<4xi32>}>
// CHECK: %[[SCALED:.*]] = tosa.mul %[[IOTA]], %[[DELTA]]
// CHECK: tosa.add %[[SCALED]], %[[START]]
// CHECK-NOT: hip.range
func.func @range_dynamic_bounds(%ctx: !hip.context, %start: tensor<i32>,
                                %limit: tensor<i32>, %delta: tensor<i32>)
    -> tensor<4xi32> attributes {rock.kernel} {
  %init = tensor.empty() : tensor<4xi32>
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i32>, tensor<i32>, tensor<i32>)
                       outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// The outlined-kernel form: the context is a ub.poison and the DPS init is a
// tensor.empty the converted op no longer reads.
// CHECK-LABEL: func.func @range_outlined_kernel
// CHECK: tosa.const{{.*}}dense<[0, 1, 2]> : tensor<3xi64>
// CHECK-NOT: hip.range
func.func @range_outlined_kernel() -> tensor<3xi64> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %start = arith.constant dense<0> : tensor<i64>
  %limit = arith.constant dense<3> : tensor<i64>
  %delta = arith.constant dense<1> : tensor<i64>
  %init = tensor.empty() : tensor<3xi64>
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i64>, tensor<i64>, tensor<i64>)
                       outs(%init : tensor<3xi64>) : tensor<3xi64>
  return %r : tensor<3xi64>
}

//===----------------------------------------------------------------------===//
// CastLike, through the hip.cast its residue becomes.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @castlike_f32_to_f16
// CHECK: tosa.cast %arg1 : (tensor<3x4xf32>) -> tensor<3x4xf16>
// CHECK-NOT: hip.cast
func.func @castlike_f32_to_f16(%ctx: !hip.context, %x: tensor<3x4xf32>,
                               %init: tensor<3x4xf16>) -> tensor<3x4xf16>
    attributes {rock.kernel} {
  %r = hip.cast(%ctx) ins(%x : tensor<3x4xf32>)
                      outs(%init : tensor<3x4xf16>) {to = 10 : i64}
                      : tensor<3x4xf16>
  return %r : tensor<3x4xf16>
}

// An identity CastLike is forwarded by simplify-onnx, so the kernel that
// reaches this pass just returns its input.
// CHECK-LABEL: func.func @castlike_identity
// CHECK: return %arg0
// CHECK-NOT: hip.cast
func.func @castlike_identity(%x: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  return %x : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected forms. A hip op this pass claims has to convert or the pass fails.
//===----------------------------------------------------------------------===//

// tosa.const needs a compile-time length, and TOSA has no arange to build one
// at runtime.
func.func @range_dynamic_length(%ctx: !hip.context, %start: tensor<i64>,
                                %limit: tensor<i64>, %delta: tensor<i64>,
                                %init: tensor<?xi64>) -> tensor<?xi64>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.range'}}
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i64>, tensor<i64>, tensor<i64>)
                       outs(%init : tensor<?xi64>) : tensor<?xi64>
  return %r : tensor<?xi64>
}

// -----

// An empty ONNX Range is valid, but a TOSA number tensor cannot be empty.
func.func @range_empty(%ctx: !hip.context, %init: tensor<0xi64>)
    -> tensor<0xi64> attributes {rock.kernel} {
  %start = arith.constant dense<5> : tensor<i64>
  %limit = arith.constant dense<2> : tensor<i64>
  %delta = arith.constant dense<1> : tensor<i64>
  // expected-error @+1 {{failed to legalize operation 'hip.range'}}
  %r = hip.range(%ctx) ins(%start, %limit, %delta
                           : tensor<i64>, tensor<i64>, tensor<i64>)
                       outs(%init : tensor<0xi64>) : tensor<0xi64>
  return %r : tensor<0xi64>
}
