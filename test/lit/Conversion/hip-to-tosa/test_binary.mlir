// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the elementwise binary hip ops that share hip.add's operand shape
// lower 1-1 to their TOSA counterparts inside a rock.kernel function.
//
// hip.add's own coverage lives in test_add.mlir. This file covers the ops
// added on top of the shared BinaryConverter template.
//
// This test validates:
// - hip.sub and hip.min map to tosa.sub and tosa.minimum
// - Size-1 dimensions rely on TOSA's implicit broadcast
// - tosa.minimum's nan_mode defaults to PROPAGATE (omitted in pretty form)
// - The pass is a no-op on functions without rock.kernel
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @sub
// CHECK: tosa.sub %arg1, %arg2 : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.sub
func.func @sub(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.sub(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @sub_broadcast
// CHECK: tosa.sub
// CHECK-NOT: hip.sub
func.func @sub_broadcast(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                         %y: tensor<1x1x32xf16>, %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.sub(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// CHECK-LABEL: func.func @min
// CHECK: tosa.minimum %arg1, %arg2 : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.min
func.func @min(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.min(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @min_broadcast
// CHECK: tosa.minimum
// CHECK-NOT: hip.min
func.func @min_broadcast(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                         %y: tensor<1x1x32xf16>, %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.min(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

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

// ResNet's first ReLU: hip.max(activation, dense<0> : tensor<f16>). The scalar
// is reshaped to 1x1x1x1, then tosa.maximum broadcasts the size-1 dims.
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
// CHECK-LABEL: func.func @sub_not_a_kernel
// CHECK: hip.sub
// CHECK-NOT: tosa.sub
func.func @sub_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                            %y: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.sub(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
