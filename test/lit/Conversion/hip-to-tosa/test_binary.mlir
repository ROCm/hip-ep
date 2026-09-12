// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the elementwise binary hip ops lower to their TOSA counterparts
// inside a rock.kernel function, so rocMLIR can absorb them into a fused
// kernel, and verify the forms the conversion rejects.
//
// All five ops share the BinaryConverter template, so the shape and broadcast
// behaviour is exercised once (through hip.add) rather than repeated per op.
// The per-op cases prove the op mapping itself, plus the operand handling
// specific to hip.mul.
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
// - add, sub, min, max and mul map to their TOSA counterparts
// - The hip context and DPS outs operand are both dropped
// - Size-1 dimensions rely on TOSA's implicit broadcast
// - Lower-rank operands are reshaped with leading 1s first
// - tosa.minimum/maximum's nan_mode defaults to PROPAGATE (omitted in the
//   pretty form)
// - tosa.mul's shift operand is materialized as a zero tensor<1xi8>
// - The pass is a no-op on functions without rock.kernel
// - Dynamic shapes, un-broadcastable dimensions and mismatched element types
//   are rejected rather than lowered to invalid TOSA
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Op mappings.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add
// CHECK: tosa.add %arg1, %arg2 : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.add
func.func @add(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

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

// tosa.mul takes a third shift operand, which hip.mul has no equivalent for:
// it is a plain multiply, so the shift is a zero tensor<1xi8>.
// CHECK-LABEL: func.func @mul
// CHECK: %[[SHIFT:.*]] = "tosa.const"{{.*}}dense<0> : tensor<1xi8>
// CHECK: tosa.mul %arg1, %arg2, %[[SHIFT]] : (tensor<2x8xf16>, tensor<2x8xf16>, tensor<1xi8>) -> tensor<2x8xf16>
// CHECK-NOT: hip.mul
func.func @mul(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.mul(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

//===----------------------------------------------------------------------===//
// Broadcasting, exercised through hip.add on behalf of the shared template.
//===----------------------------------------------------------------------===//

// Size-1 dims are TOSA-broadcastable, so this still maps 1-1.
// CHECK-LABEL: func.func @add_broadcast
// CHECK: tosa.add
// CHECK-NOT: hip.add
func.func @add_broadcast(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                         %y: tensor<1x1x32xf16>, %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// hip rank-extends the way ONNX/NumPy do, so a lower-rank operand is reshaped
// with leading 1s before tosa.add broadcasts it.
// CHECK-LABEL: func.func @add_rank_extending_broadcast
// CHECK: tosa.reshape
// CHECK: tosa.add
// CHECK-NOT: hip.add
func.func @add_rank_extending_broadcast(%ctx: !hip.context,
                                        %x: tensor<1x128x32xf16>,
                                        %y: tensor<32xf16>,
                                        %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<32xf16>)
                     outs(%init : tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// A rank-0 scalar is the same rank mismatch, and is how a bias add shows up
// after onnx-to-hip lowering.
// CHECK-LABEL: func.func @add_scalar_operand
// CHECK: tosa.reshape
// CHECK: tosa.add
// CHECK-NOT: hip.add
func.func @add_scalar_operand(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                              %y: tensor<f16>, %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<f16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     -> tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

//===----------------------------------------------------------------------===//
// Cases that are not covered by the shared shape handling above.
//===----------------------------------------------------------------------===//

// ResNet's first ReLU: hip.max(activation, dense<0> : tensor<f16>). Kept as
// its own case because it is the shape that motivated rank equalization.
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

// A zero shift is what makes the integer case a plain multiply rather than a
// rescaled one.
// CHECK-LABEL: func.func @mul_integer
// CHECK: dense<0> : tensor<1xi8>
// CHECK: tosa.mul
// CHECK-NOT: hip.mul
func.func @mul_integer(%ctx: !hip.context, %x: tensor<4x4xi32>,
                       %y: tensor<4x4xi32>, %init: tensor<4x4xi32>)
    -> tensor<4x4xi32> attributes {rock.kernel} {
  %r = hip.mul(%ctx) ins(%x, %y : tensor<4x4xi32>, tensor<4x4xi32>)
                     outs(%init : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %r : tensor<4x4xi32>
}

// The shift is excluded from tosa.mul's same-rank verification, so equalizing
// the two data operands is all that is needed.
// CHECK-LABEL: func.func @mul_scalar_operand
// CHECK: tosa.reshape
// CHECK: tosa.mul
// CHECK-NOT: hip.mul
func.func @mul_scalar_operand(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                              %y: tensor<f16>, %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  %r = hip.mul(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<f16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     -> tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

//===----------------------------------------------------------------------===//
// The rock.kernel guard, which is a property of the pass rather than of any
// one op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @add_not_a_kernel
// CHECK: hip.add
// CHECK-NOT: tosa.add
func.func @add_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                            %y: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected forms, one per chunk. Exercised through hip.add on behalf of the
// shared template. The pass marks the whole hip dialect illegal, so these fail
// legalization rather than surviving in the output.
//===----------------------------------------------------------------------===//

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %y: tensor<?x8xf16>, %init: tensor<?x8xf16>)
    -> tensor<?x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<?x8xf16>, tensor<?x8xf16>)
                     outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// Rank equalization prepends 1s, so tensor<8xf16> becomes 1x1x1x8. The
// trailing 8 still cannot broadcast to 112, which the post-equalization check
// catches rather than emitting invalid TOSA.
func.func @incompatible_broadcast(%ctx: !hip.context,
                                  %x: tensor<1x64x112x112xf16>,
                                  %y: tensor<8xf16>,
                                  %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<8xf16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     -> tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// -----

// A mismatched element type is not a broadcast question at all.
func.func @element_type_mismatch(%ctx: !hip.context, %x: tensor<2x8xf16>,
                                 %y: tensor<2x8xf32>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.add'}}
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf32>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
