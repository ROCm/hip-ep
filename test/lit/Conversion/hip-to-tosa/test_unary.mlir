// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the elementwise unary hip ops lower 1-1 to their TOSA counterparts
// inside a rock.kernel function, so rocMLIR can absorb them into a fused
// kernel, and verify the forms the conversion rejects.
//
// FILE LAYOUT:
// Everything that converts lives in the first --split-input-file chunk, so it
// is one module and therefore also covers several ops converting in a single
// pass run. Each rejected form then gets its own chunk: full conversion turns
// a rejection into a pass failure that aborts the run for the whole module, so
// sharing a chunk would let one rejection mask the cases after it. A failing
// chunk contributes no output, while the chunks that convert still print for
// FileCheck.
//
// This test validates:
// - Each of the twelve unary ops maps to its TOSA counterpart
// - hip.cast maps to tosa.cast, or rocMLIR tosa.custom for float->int / unsigned
// - The hip context and the DPS `y` operand are both dropped
// - hip.neg picks up tosa.negate's materialized zero-point operands
// - The pass is a no-op on functions without rock.kernel
// - Dynamic shapes, operands that would have to broadcast, and integer
//   operands to the float-only ops are all rejected
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

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

// hip.cast is same-shape; float->float and int->float are tosa.cast.
// CHECK-LABEL: func.func @cast_f32_to_f16
// CHECK: tosa.cast %arg1 : (tensor<2x8xf32>) -> tensor<2x8xf16>
// CHECK-NOT: hip.cast
func.func @cast_f32_to_f16(%ctx: !hip.context, %x: tensor<2x8xf32>,
                            %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.cast(%ctx) ins(%x : tensor<2x8xf32>)
                      outs(%init : tensor<2x8xf16>) {to = 10 : i64}
                      : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @cast_i8_to_f32
// CHECK: tosa.cast %arg1 : (tensor<4xi8>) -> tensor<4xf32>
// CHECK-NOT: hip.cast
func.func @cast_i8_to_f32(%ctx: !hip.context, %x: tensor<4xi8>,
                           %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.cast(%ctx) ins(%x : tensor<4xi8>)
                      outs(%init : tensor<4xf32>) {to = 1 : i64}
                      : tensor<4xf32>
  return %r : tensor<4xf32>
}

// rock-tosa-to-elementwise rejects tosa.cast float->int; emit rocMLIR custom.
// CHECK-LABEL: func.func @cast_f32_to_i8
// CHECK: tosa.custom %arg1 {{.*}}operator_name = "fp_to_int_cast"
// CHECK-NOT: hip.cast
func.func @cast_f32_to_i8(%ctx: !hip.context, %x: tensor<2x8xf32>,
                           %init: tensor<2x8xi8>) -> tensor<2x8xi8>
    attributes {rock.kernel} {
  %r = hip.cast(%ctx) ins(%x : tensor<2x8xf32>)
                      outs(%init : tensor<2x8xi8>) {to = 3 : i64}
                      : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// CHECK-LABEL: func.func @cast_ui8_to_f32
// CHECK: tosa.custom %arg1 {{.*}}operator_name = "unsigned_cast"
// CHECK-NOT: hip.cast
func.func @cast_ui8_to_f32(%ctx: !hip.context, %x: tensor<4xui8>,
                            %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.cast(%ctx) ins(%x : tensor<4xui8>)
                      outs(%init : tensor<4xf32>) {to = 1 : i64}
                      : tensor<4xf32>
  return %r : tensor<4xf32>
}

// CHECK-LABEL: func.func @cast_outlined_kernel
// CHECK: tosa.cast
// CHECK-NOT: hip.cast
func.func @cast_outlined_kernel(%x: tensor<2x8xf32>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.cast(%ctx) ins(%x : tensor<2x8xf32>)
                      outs(%init : tensor<2x8xf16>) {to = 10 : i64}
                      : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// The shape hip-fuse-rocmlir actually produces: the !hip.context is a
// ub.poison materialized inside the outlined kernel rather than a block
// argument. The ConversionTarget marks ub.poison legal so it survives as dead
// IR for the canonicalizer to drop, instead of failing legalization.
// CHECK-LABEL: func.func @abs_outlined_kernel
// CHECK: tosa.abs
// CHECK-NOT: hip.abs
func.func @abs_outlined_kernel(%x: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.abs(%ctx) ins(%x : tensor<2x8xf16>)
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

// -----

//===----------------------------------------------------------------------===//
// Rejected forms, one per chunk. The pass marks the whole hip dialect illegal,
// so these fail legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.abs'}}
  %r = hip.abs(%ctx) ins(%x : tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// TOSA unary ops carry SameOperandsAndResultShape, so an operand that would
// have to broadcast to the result is not a 1-1 mapping. Unlike the binary ops
// there is no size-1 broadcast to fall back on.
func.func @shape_mismatch(%ctx: !hip.context, %x: tensor<1x1x32xf16>,
                          %init: tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.exp'}}
  %r = hip.exp(%ctx) ins(%x : tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// -----

// tosa.sin takes Tosa_FloatTensor, so an integer operand would fail the TOSA
// verifier. Reject it here instead of emitting invalid TOSA.
func.func @integer_operand(%ctx: !hip.context, %x: tensor<4xi32>,
                           %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.sin'}}
  %r = hip.sin(%ctx) ins(%x : tensor<4xi32>)
                     outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

func.func @cast_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                               %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.cast'}}
  %r = hip.cast(%ctx) ins(%x : tensor<?x8xf32>)
                      outs(%init : tensor<?x8xf16>) {to = 10 : i64}
                      : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}
