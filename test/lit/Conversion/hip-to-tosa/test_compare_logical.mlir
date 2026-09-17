// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the comparison, logical and sign hip ops lower to TOSA inside a
// rock.kernel function, so rocMLIR can absorb them into a fused kernel, and
// verify the forms the conversion rejects.
//
// These are the hip ops whose operand and result element types differ, or
// whose TOSA spelling is not a single op:
//   - hip.equal / hip.less produce i1 from operands of the compared type, so
//     they use ComparisonConverter rather than BinaryConverter.
//   - hip.and / hip.or / hip.not are i1 throughout and reuse the shared
//     templates with the BoolOnly gate.
//   - hip.sign has no TOSA counterpart and expands to nested selects.
//
// hip.greater, hip.greater_or_equal and hip.less_or_equal are deliberately
// absent: no such hip ops exist. OnnxToHip decomposes onnx.Greater into
// hip.less(B, A) and the two or-equal forms into hip.not(hip.less(..)), so
// they reach this pass as the ops covered below.
//
// FILE LAYOUT:
// Everything that converts lives in the first --split-input-file chunk, so it
// is one module and therefore also covers several ops converting in a single
// pass run. Each rejected form then gets its own chunk: a rejection aborts the
// run for the whole module, so sharing a chunk would let one rejection mask
// the cases after it. A failing chunk contributes no output, while the chunks
// that convert still print for FileCheck.
//
// This test validates:
// - hip.equal maps to tosa.equal with its operands in order
// - hip.less maps to tosa.greater with its operands swapped
// - hip.and / hip.or / hip.not map to the tosa.logical_* ops
// - hip.sign expands to select(x > 0, 1, select(0 > x, -1, 0))
// - The i1 result of a comparison does not block broadcasting of its operands
// - The hip context and the DPS outs operand are both dropped
// - The pass is a no-op on functions without rock.kernel
// - Dynamic shapes, un-broadcastable operands, mismatched operand element
//   types and non-i1 operands to the logical ops are all rejected
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Op mappings.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @equal
// CHECK: tosa.equal %arg1, %arg2 : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
// CHECK-NOT: hip.equal
func.func @equal(%ctx: !hip.context, %x: tensor<2x8xf32>, %y: tensor<2x8xf32>,
                 %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                       outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// TOSA has no `less`, so `x < y` is emitted as `y > x`: the operands appear in
// the opposite order to the hip op's.
// CHECK-LABEL: func.func @less
// CHECK: tosa.greater %arg2, %arg1 : (tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xi1>
// CHECK-NOT: hip.less
func.func @less(%ctx: !hip.context, %x: tensor<2x8xf32>, %y: tensor<2x8xf32>,
                %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// Integer operands take the same path; only the operand element type changes.
// CHECK-LABEL: func.func @less_i32
// CHECK: tosa.greater %arg2, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
func.func @less_i32(%ctx: !hip.context, %x: tensor<4xi32>, %y: tensor<4xi32>,
                    %init: tensor<4xi1>) -> tensor<4xi1>
    attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<4xi32>, tensor<4xi32>)
                      outs(%init : tensor<4xi1>) : tensor<4xi1>
  return %r : tensor<4xi1>
}

// CHECK-LABEL: func.func @and
// CHECK: tosa.logical_and %arg1, %arg2 : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.and
func.func @and(%ctx: !hip.context, %x: tensor<2x8xi1>, %y: tensor<2x8xi1>,
               %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xi1>, tensor<2x8xi1>)
                     outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// CHECK-LABEL: func.func @or
// CHECK: tosa.logical_or %arg1, %arg2 : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.or
func.func @or(%ctx: !hip.context, %x: tensor<2x8xi1>, %y: tensor<2x8xi1>,
              %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.or(%ctx) ins(%x, %y : tensor<2x8xi1>, tensor<2x8xi1>)
                    outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// CHECK-LABEL: func.func @not
// CHECK: tosa.logical_not %arg1 : (tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.not
func.func @not(%ctx: !hip.context, %x: tensor<2x8xi1>,
               %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.not(%ctx) ins(%x : tensor<2x8xi1>)
                     outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

//===----------------------------------------------------------------------===//
// Broadcasting, which for a comparison has to be checked against the operand
// element type rather than the i1 result.
//===----------------------------------------------------------------------===//

// A size-1 dimension is left to TOSA's implicit broadcast, so no reshape is
// emitted and the operands reach tosa.greater at their original types.
// CHECK-LABEL: func.func @less_broadcast_size1
// CHECK-NOT: tosa.reshape
// CHECK: tosa.greater %arg2, %arg1 : (tensor<2x8xf32>, tensor<2x1xf32>) -> tensor<2x8xi1>
func.func @less_broadcast_size1(%ctx: !hip.context, %x: tensor<2x1xf32>,
                                %y: tensor<2x8xf32>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x1xf32>, tensor<2x8xf32>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// A lower-rank operand is rank-extended with leading 1s first. The reshape
// carries the operand element type, not i1.
// CHECK-LABEL: func.func @equal_broadcast_rank
// CHECK: %[[R:.*]] = tosa.reshape %arg2, %{{.*}} : (tensor<8xf32>, !tosa.shape<2>) -> tensor<1x8xf32>
// CHECK: tosa.equal %arg1, %[[R]] : (tensor<2x8xf32>, tensor<1x8xf32>) -> tensor<2x8xi1>
func.func @equal_broadcast_rank(%ctx: !hip.context, %x: tensor<2x8xf32>,
                                %y: tensor<8xf32>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<8xf32>)
                       outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

//===----------------------------------------------------------------------===//
// hip.sign, which expands rather than mapping 1-1.
//===----------------------------------------------------------------------===//

// select(x > 0, 1, select(0 > x, -1, 0)). Three constants rather than four:
// the inner select's else value reuses the zero both comparisons already need.
// The checks follow emission order, in which each constant is materialized at
// its first use and so is interleaved with the comparisons.
// CHECK-LABEL: func.func @sign
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x8xf32>}>
// CHECK: %[[POS:.*]] = tosa.greater %arg1, %[[ZERO]]
// CHECK: %[[ISNEG:.*]] = tosa.greater %[[ZERO]], %arg1
// CHECK: %[[NEG:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<2x8xf32>}>
// CHECK: %[[INNER:.*]] = tosa.select %[[ISNEG]], %[[NEG]], %[[ZERO]]
// CHECK: %[[ONE:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x8xf32>}>
// CHECK: tosa.select %[[POS]], %[[ONE]], %[[INNER]]
// CHECK-NOT: hip.sign
func.func @sign(%ctx: !hip.context, %x: tensor<2x8xf32>,
                %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<2x8xf32>)
                      outs(%init : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// ONNX Sign is defined over signed integers too, where the constants are
// integer rather than float splats.
// CHECK-LABEL: func.func @sign_i32
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<4xi32>}>
// CHECK: %[[POS:.*]] = tosa.greater %arg1, %[[ZERO]]
// CHECK: %[[ISNEG:.*]] = tosa.greater %[[ZERO]], %arg1
// CHECK: %[[NEG:.*]] = "tosa.const"() <{values = dense<-1> : tensor<4xi32>}>
// CHECK: %[[INNER:.*]] = tosa.select %[[ISNEG]], %[[NEG]], %[[ZERO]]
// CHECK: %[[ONE:.*]] = "tosa.const"() <{values = dense<1> : tensor<4xi32>}>
// CHECK: tosa.select %[[POS]], %[[ONE]], %[[INNER]]
func.func @sign_i32(%ctx: !hip.context, %x: tensor<4xi32>,
                    %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<4xi32>)
                      outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

//===----------------------------------------------------------------------===//
// The rock.kernel guard, which is a property of the pass rather than of any
// one op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @less_not_a_kernel
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf32>,
                             %y: tensor<2x8xf32>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected forms, one per chunk. The pass marks each of these ops illegal, so
// a rejection fails legalization rather than leaving the hip op in place.
//===----------------------------------------------------------------------===//

// Dynamic shapes give the pattern no static shape to reason about.
func.func @less_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                              %y: tensor<?x8xf32>, %init: tensor<?x8xi1>)
    -> tensor<?x8xi1> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.less'}}
  %r = hip.less(%ctx) ins(%x, %y : tensor<?x8xf32>, tensor<?x8xf32>)
                      outs(%init : tensor<?x8xi1>) : tensor<?x8xi1>
  return %r : tensor<?x8xi1>
}

// -----

// Rank equalization prepends 1s, so tensor<4xf32> becomes 1x4. The trailing 4
// still cannot broadcast to 8, which the post-equalization check catches
// rather than emitting invalid TOSA.
func.func @less_incompatible_broadcast(%ctx: !hip.context,
                                       %x: tensor<2x8xf32>, %y: tensor<4xf32>,
                                       %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.less'}}
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<4xf32>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// Both TOSA comparisons carry SameOperandsElementType, so operands of
// different types have no valid spelling even though the result is i1 either
// way.
func.func @equal_element_type_mismatch(%ctx: !hip.context,
                                       %x: tensor<2x8xf32>,
                                       %y: tensor<2x8xf16>,
                                       %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.equal'}}
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// A comparison whose result is not i1 is malformed; naming it here is clearer
// than letting the TOSA verifier reject the op this pass synthesized.
func.func @equal_non_i1_result(%ctx: !hip.context, %x: tensor<2x8xf32>,
                               %y: tensor<2x8xf32>, %init: tensor<2x8xi8>)
    -> tensor<2x8xi8> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.equal'}}
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                       outs(%init : tensor<2x8xi8>) : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// -----

// tosa.logical_and takes Tosa_I1Tensor, so a wider integer is rejected by the
// BoolOnly gate instead of being replaced with an op that fails verification.
func.func @and_non_i1(%ctx: !hip.context, %x: tensor<2x8xi8>,
                      %y: tensor<2x8xi8>, %init: tensor<2x8xi8>)
    -> tensor<2x8xi8> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.and'}}
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xi8>, tensor<2x8xi8>)
                     outs(%init : tensor<2x8xi8>) : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// -----

// The same gate on the unary side, for tosa.logical_not.
func.func @not_non_i1(%ctx: !hip.context, %x: tensor<2x8xi8>,
                      %init: tensor<2x8xi8>) -> tensor<2x8xi8>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.not'}}
  %r = hip.not(%ctx) ins(%x : tensor<2x8xi8>)
                     outs(%init : tensor<2x8xi8>) : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// -----

// i1 cannot represent the -1 that Sign needs for negative inputs.
func.func @sign_i1(%ctx: !hip.context, %x: tensor<2x8xi1>,
                   %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.sign'}}
  %r = hip.sign(%ctx) ins(%x : tensor<2x8xi1>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}
