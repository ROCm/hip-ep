// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the elementwise unary hip ops lower 1-1 to their TOSA counterparts
// inside a rock.kernel function, so rocMLIR can absorb them into a fused
// kernel.
//
// This test validates:
// - Each of the twelve unary ops maps to its TOSA counterpart
// - The hip context and the DPS `y` operand are both dropped
// - hip.neg picks up tosa.negate's materialized zero-point operands
// - The pass is a no-op on functions without rock.kernel
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @abs
// CHECK: tosa.abs %arg1 : (tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.abs
func.func @abs(%ctx: !hip.context, %x: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.abs(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// tosa.negate carries input1_zp/output_zp operands that the quant-info builder
// materializes as tosa.const, so this op expands to more than one op.
// CHECK-LABEL: func.func @neg
// CHECK: tosa.const
// CHECK: tosa.negate
// CHECK-NOT: hip.neg
func.func @neg(%ctx: !hip.context, %x: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.neg(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @ceil
// CHECK: tosa.ceil
// CHECK-NOT: hip.ceil
func.func @ceil(%ctx: !hip.context, %x: tensor<4xf32>,
                %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.ceil(%ctx) ins(%x : tensor<4xf32>)
                      outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @floor
// CHECK: tosa.floor
// CHECK-NOT: hip.floor
func.func @floor(%ctx: !hip.context, %x: tensor<4xf32>,
                 %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.floor(%ctx) ins(%x : tensor<4xf32>)
                       outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @exp
// CHECK: tosa.exp
// CHECK-NOT: hip.exp
func.func @exp(%ctx: !hip.context, %x: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.exp(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @log
// CHECK: tosa.log
// CHECK-NOT: hip.log
func.func @log(%ctx: !hip.context, %x: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.log(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @sin
// CHECK: tosa.sin
// CHECK-NOT: hip.sin
func.func @sin(%ctx: !hip.context, %x: tensor<4xf32>,
               %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.sin(%ctx) ins(%x : tensor<4xf32>)
                     outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @cos
// CHECK: tosa.cos
// CHECK-NOT: hip.cos
func.func @cos(%ctx: !hip.context, %x: tensor<4xf32>,
               %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.cos(%ctx) ins(%x : tensor<4xf32>)
                     outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @tanh
// CHECK: tosa.tanh
// CHECK-NOT: hip.tanh
func.func @tanh(%ctx: !hip.context, %x: tensor<2x8xf16>,
                %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.tanh(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @erf
// CHECK: tosa.erf
// CHECK-NOT: hip.erf
func.func @erf(%ctx: !hip.context, %x: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.erf(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// A real activation shape, to confirm rank does not matter for unary ops.
// CHECK-LABEL: func.func @sigmoid
// CHECK: tosa.sigmoid
// CHECK-NOT: hip.sigmoid
func.func @sigmoid(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                   %init: tensor<1x64x112x112xf16>) -> tensor<1x64x112x112xf16>
    attributes {rock.kernel} {
  %r = hip.sigmoid(%ctx) ins(%x : tensor<1x64x112x112xf16>)
                         outs(%init : tensor<1x64x112x112xf16>)
                         : tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// CHECK-LABEL: func.func @reciprocal
// CHECK: tosa.reciprocal
// CHECK-NOT: hip.reciprocal
func.func @reciprocal(%ctx: !hip.context, %x: tensor<2x8xf16>,
                      %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.reciprocal(%ctx) ins(%x : tensor<2x8xf16>)
                            outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// The pass early-returns unless the function is an outlined kernel.
// CHECK-LABEL: func.func @abs_not_a_kernel
// CHECK: hip.abs
// CHECK-NOT: tosa.abs
func.func @abs_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                            %init: tensor<2x8xf16>) -> tensor<2x8xf16> {
  %r = hip.abs(%ctx) ins(%x : tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
