// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.max is lowered to tosa.maximum inside a rock.kernel function.
//
// This test validates:
// - Same-rank operands map directly to tosa.maximum
// - Size-1 dimensions rely on TOSA's implicit broadcast
// - ResNet ReLU: rank-0 scalar (tensor<f16>) is reshaped then broadcast
// - The hip context and DPS outs operand are both dropped
// - nan_mode is TOSA's default PROPAGATE (omitted in pretty form)
// - The pass is a no-op on functions without rock.kernel
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @max
// CHECK: tosa.maximum %arg1, %arg2 : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.max
func.func @max(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.max(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// Size-1 dims are TOSA-broadcastable, so this still maps 1-1.
// CHECK-LABEL: func.func @max_broadcast
// CHECK: tosa.maximum
// CHECK-NOT: hip.max
func.func @max_broadcast(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                         %y: tensor<1x1x32xf16>, %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.max(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// ResNet first ReLU: hip.max(activation, dense<0> : tensor<f16>).
// TOSA needs matching rank, so the scalar is reshaped to 1x1x1x1.
// CHECK-LABEL: func.func @max_relu_scalar
// CHECK: tosa.reshape
// CHECK: tosa.maximum
// CHECK-NOT: hip.max
func.func @max_relu_scalar(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                          %zero: tensor<f16>, %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  %r = hip.max(%ctx) ins(%x, %zero : tensor<1x64x112x112xf16>, tensor<f16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     : tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// The pass early-returns unless the function is an outlined kernel.
// CHECK-LABEL: func.func @max_not_a_kernel
// CHECK: hip.max
// CHECK-NOT: tosa.maximum
func.func @max_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                           %y: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.max(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
