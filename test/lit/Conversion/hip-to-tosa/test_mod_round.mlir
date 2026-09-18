// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that hip.round and hip.mod, the lowerings of ONNX Round and ONNX Mod,
// become TOSA inside a rock.kernel function. Both are already in FuseROCMlir's
// pointwise list, so either one sitting behind a matmul is outlined into the
// kernel whether or not this pass can lower it; left unconverted it reaches
// rocMLIR as a hip op that rocMLIR cannot compile.
//
// TOSA has an op for neither, so each is spelled out. Both expansions are
// exact -- nothing here approximates.
//
// hip.round is ONNX Round, which breaks ties to even, so floor(x + 0.5) is the
// wrong rounding on every tie. The expansion takes floor(x), then steps up when
// the fraction exceeds a half and, on the tie, only when the floor is odd.
//
// hip.mod rests on tosa.intdiv truncating towards zero: lhs - (lhs / rhs) * rhs
// is C's %, whose sign follows the dividend, which is ONNX Mod's fmod = 1
// exactly. The default fmod = 0 wants the divisor's sign instead, and the two
// differ by one divisor precisely where the remainder and divisor signs
// disagree and the remainder is not zero.
//
// Both expansions combine their predicates with the bitwise ops rather than the
// logical ones, which coincide on i1. rocMLIR's RockTosaToElementwise has
// patterns for tosa.bitwise_and/_or/_xor and none for any tosa.logical_*, and
// it marks every surviving tosa op illegal, so the logical spelling converts
// cleanly here and then kills the kernel it was meant to enable.
//
// COVERAGE:
// - Round emits the floor/compare/select expansion, with the tie broken on the
//   parity of the floor rather than on the fraction alone
// - Round steps up with tosa.ceil, so a value in [-0.5, 0) keeps its signed
//   zero as nearbyintf does in the round runtime
// - Round converts for f32, f16 and bf16 alike, and names f64, which ONNX
//   allows and TOSA has no tensor type for
// - Mod divides by a divisor with -1 substituted away, since the most negative
//   value over -1 is undefined behaviour once tosa.intdiv becomes an sdiv
// - Neither expansion emits a tosa.logical_* op
// - Mod with the default fmod = 0 emits the remainder plus the divisor-sign
//   correction; fmod = 1 stops at the bare remainder with no select
// - Signless i32 and i64, the only widths tosa.intdiv accepts, both convert
// - A lower-rank divisor is reshaped with leading 1s first, since TOSA
//   broadcasts size-1 dimensions only once both operands carry the result rank
// - Each op converts behind a matmul anchor, the shape hip-fuse-rocmlir makes
// - Float mod is rejected with the reason named: fmod needs the exact truncated
//   quotient, which a reciprocal and a multiply cannot supply
// - Integer widths tosa.intdiv cannot take are rejected with the type named
// - The pass is a no-op on functions without rock.kernel
// - Dynamic shapes are rejected, matching the sibling elementwise ops
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// hip.round.
//===----------------------------------------------------------------------===//

// The full expansion. %[[F]] is the integer below x and %[[D]] the fraction;
// the result steps up when the fraction exceeds a half, or when it is exactly a
// half and %[[F]] is odd. Oddness is the floor failing to survive a halving and
// a doubling, which is why the second floor and the multiply by two are here
// rather than a cheaper test on the fraction; halving drops the low bit, so an
// odd floor comes back one smaller and %[[ODD]] can read that as a compare.
//
// The step-up value is tosa.ceil rather than %[[F]] + 1 so that an input in
// [-0.5, 0) keeps the sign of its zero, as nearbyintf does in the round
// runtime. The two agree wherever the branch is taken, since it is taken only
// on a non-zero fraction.
// CHECK-LABEL: func.func @round_f32
// CHECK-DAG: %[[HALF:.*]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<4xf32>}>
// CHECK-DAG: %[[TWO:.*]] = "tosa.const"() <{values = dense<2.000000e+00> : tensor<4xf32>}>
// CHECK: %[[F:.*]] = tosa.floor %arg1
// CHECK: %[[D:.*]] = tosa.sub %arg1, %[[F]]
// CHECK: %[[ABOVE:.*]] = tosa.greater %[[D]], %[[HALF]]
// CHECK: %[[TIE:.*]] = tosa.equal %[[D]], %[[HALF]]
// CHECK: %[[HALVED:.*]] = tosa.mul %[[F]], %[[HALF]]
// CHECK: %[[FH:.*]] = tosa.floor %[[HALVED]]
// CHECK: %[[BACK:.*]] = tosa.mul %[[FH]], %[[TWO]]
// CHECK: %[[ODD:.*]] = tosa.greater %[[F]], %[[BACK]]
// CHECK: %[[TIEUP:.*]] = tosa.bitwise_and %[[TIE]], %[[ODD]]
// CHECK: %[[UP:.*]] = tosa.bitwise_or %[[ABOVE]], %[[TIEUP]]
// CHECK: %[[NEXT:.*]] = tosa.ceil %arg1
// CHECK: tosa.select %[[UP]], %[[NEXT]], %[[F]]
// CHECK-NOT: hip.round
// The predicates are combined bitwise, never logically: rocMLIR's
// RockTosaToElementwise has no pattern for any tosa.logical_* op and marks
// every surviving tosa op illegal, so the logical spelling would convert here
// and then kill the kernel this conversion exists to enable.
// CHECK-NOT: tosa.logical
func.func @round_f32(%ctx: !hip.context, %x: tensor<4xf32>,
                     %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.round(%ctx) ins(%x : tensor<4xf32>)
                       outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// Nothing in the expansion is f32-specific; the constants take the operand's
// own element type.
// CHECK-LABEL: func.func @round_f16
// CHECK: %[[F:.*]] = tosa.floor %arg1 : (tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK: tosa.select %{{.*}} : (tensor<2x8xi1>, tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.round
func.func @round_f16(%ctx: !hip.context, %x: tensor<2x8xf16>,
                     %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.round(%ctx) ins(%x : tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

//===----------------------------------------------------------------------===//
// hip.mod.
//===----------------------------------------------------------------------===//

// fmod = 0, the ONNX default: the truncated remainder, then the correction that
// moves its sign onto the divisor. The correction fires only where the two
// signs disagree and the remainder is non-zero -- an exact divide must not
// collect a spurious divisor.
//
// The divide runs against %[[DIV]], which is the divisor with -1 replaced by 1,
// because the most negative value over -1 overflows and tosa.intdiv becomes an
// LLVM sdiv, where that pair is undefined behaviour rather than a wrong value.
// Every remainder by -1 is zero under both fmod rules and dividing by 1 gives
// exactly that, so the substitution costs nothing but the overflow.
// CHECK-LABEL: func.func @mod_i32
// CHECK-DAG: %[[MONE:.*]] = "tosa.const"() <{values = dense<-1> : tensor<4xi32>}>
// CHECK-DAG: %[[ONE:.*]] = "tosa.const"() <{values = dense<1> : tensor<4xi32>}>
// CHECK: %[[ISMONE:.*]] = tosa.equal %arg2, %[[MONE]]
// CHECK: %[[DIV:.*]] = tosa.select %[[ISMONE]], %[[ONE]], %arg2
// CHECK: %[[Q:.*]] = tosa.intdiv %arg1, %[[DIV]]
// CHECK: %[[P:.*]] = tosa.mul %[[Q]], %[[DIV]]
// CHECK: %[[R:.*]] = tosa.sub %arg1, %[[P]]
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<4xi32>}>
// CHECK: %[[BNEG:.*]] = tosa.greater %[[ZERO]], %arg2
// CHECK: %[[RNEG:.*]] = tosa.greater %[[ZERO]], %[[R]]
// CHECK: %[[DIFF:.*]] = tosa.bitwise_xor %[[RNEG]], %[[BNEG]]
// CHECK: %[[TRUE:.*]] = "tosa.const"() <{values = dense<true> : tensor<4xi1>}>
// CHECK: %[[ISZERO:.*]] = tosa.equal %[[R]], %[[ZERO]]
// CHECK: %[[NONZERO:.*]] = tosa.bitwise_xor %[[ISZERO]], %[[TRUE]]
// CHECK: %[[ADJ:.*]] = tosa.bitwise_and %[[DIFF]], %[[NONZERO]]
// CHECK: %[[SHIFTED:.*]] = tosa.add %[[R]], %arg2
// CHECK: tosa.select %[[ADJ]], %[[SHIFTED]], %[[R]]
// CHECK-NOT: hip.mod
// CHECK-NOT: tosa.logical
func.func @mod_i32(%ctx: !hip.context, %a: tensor<4xi32>, %b: tensor<4xi32>,
                   %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xi32>, tensor<4xi32>)
                     outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// fmod = 1 asks for the dividend's sign, which is what the truncated remainder
// already has, so the correction is absent entirely rather than emitted and
// folded later.
// CHECK-LABEL: func.func @mod_fmod_i32
// CHECK: %[[DIV:.*]] = tosa.select %{{.*}}, %{{.*}}, %arg2
// CHECK: %[[Q:.*]] = tosa.intdiv %arg1, %[[DIV]]
// CHECK: %[[P:.*]] = tosa.mul %[[Q]], %[[DIV]]
// CHECK: tosa.sub %arg1, %[[P]]
// The sign correction is absent, so the only select left is the overflow
// substitution matched above.
// CHECK-NOT: tosa.select
// CHECK-NOT: tosa.bitwise_xor
// CHECK-NOT: hip.mod
func.func @mod_fmod_i32(%ctx: !hip.context, %a: tensor<4xi32>,
                        %b: tensor<4xi32>, %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xi32>, tensor<4xi32>)
                     outs(%init : tensor<4xi32>) {fmod = 1 : i64} : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// i64 is the other half of tosa.intdiv's Tosa_Int32Or64Tensor operand type.
// The i64 multiply in the expansion lowers too: rocMLIR's MulConverter takes
// any integer element type on a zero shift, which is the shift this pass emits.
// CHECK-LABEL: func.func @mod_i64
// CHECK: tosa.intdiv %arg1, %{{.*}} : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
// CHECK: tosa.mul %{{.*}} : (tensor<4xi64>, tensor<4xi64>, tensor<1xi8>) -> tensor<4xi64>
// CHECK-NOT: hip.mod
func.func @mod_i64(%ctx: !hip.context, %a: tensor<4xi64>, %b: tensor<4xi64>,
                   %init: tensor<4xi64>) -> tensor<4xi64>
    attributes {rock.kernel} {
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xi64>, tensor<4xi64>)
                     outs(%init : tensor<4xi64>) : tensor<4xi64>
  return %r : tensor<4xi64>
}

// -----

// A rank-0 divisor is reshaped to the result's rank first, the same way the
// binary siblings do it, and every op in the expansion then broadcasts it.
// CHECK-LABEL: func.func @mod_rank_extending
// CHECK: %[[SHAPE:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>}
// CHECK: %[[B:.*]] = tosa.reshape %arg2, %[[SHAPE]] : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
// CHECK: %[[DIV:.*]] = tosa.select %{{.*}}, %{{.*}}, %[[B]] : (tensor<4x8xi1>, tensor<4x8xi32>, tensor<1x1xi32>) -> tensor<4x8xi32>
// CHECK: tosa.intdiv %arg1, %[[DIV]] : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi32>
// CHECK-NOT: hip.mod
func.func @mod_rank_extending(%ctx: !hip.context, %a: tensor<4x8xi32>,
                              %b: tensor<i32>, %init: tensor<4x8xi32>)
    -> tensor<4x8xi32> attributes {rock.kernel} {
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4x8xi32>, tensor<i32>)
                     outs(%init : tensor<4x8xi32>) : tensor<4x8xi32>
  return %r : tensor<4x8xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Behind an anchor, in the shape hip-fuse-rocmlir produces.
//===----------------------------------------------------------------------===//

// The reason for converting these at all. Both ops are in FuseROCMlir's
// pointwise list, so a round after a matmul is outlined into the kernel whether
// or not it can be lowered; left unconverted it reaches rocMLIR as a hip op fed
// by a ub.poison context, neither of which rocMLIR knows.
// CHECK-LABEL: func.func @rocMlir0
// CHECK: ub.poison : !hip.context
// CHECK: %[[MM:.*]] = tosa.matmul
// CHECK: tosa.floor
// CHECK: tosa.select
// CHECK-NOT: hip.round
// CHECK-NOT: hip.matmul
func.func @rocMlir0(%a: tensor<8x64x64xf32>, %w: tensor<64x64xf32>)
    -> tensor<8x64x64xf32> attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %m_init = tensor.empty() : tensor<8x64x64xf32>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<8x64x64xf32>, tensor<64x64xf32>)
                        outs(%m_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  %r_init = tensor.empty() : tensor<8x64x64xf32>
  %r = hip.round(%ctx) ins(%m : tensor<8x64x64xf32>)
                       outs(%r_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  return %r : tensor<8x64x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Element types with no TOSA spelling.
//===----------------------------------------------------------------------===//

// Float mod is the one form that is declined on numerical grounds rather than
// for want of an op. fmod needs the truncated quotient exactly, and a
// reciprocal and a multiply cannot supply it: an error of one ulp in lhs/rhs
// moves the truncation across an integer boundary, and the result is then wrong
// by a whole divisor rather than by an ulp. The quotient need not even be
// representable -- fmod(1e30, 3) asks for a truncation no f32 holds -- which is
// why libm reduces iteratively and why no fixed op sequence stands in for it.
func.func @mod_f32(%ctx: !hip.context, %a: tensor<4xf32>, %b: tensor<4xf32>,
                   %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  // expected-error @+2 {{hip.mod has no TOSA spelling for element type 'f32'}}
  // expected-error @+1 {{failed to legalize operation 'hip.mod'}}
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
                     outs(%init : tensor<4xf32>) {fmod = 1 : i64} : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// tosa.intdiv takes signless i32 and i64 only, and nothing upstream narrows
// what hip.mod can carry, so the narrower widths reach this pass intact and are
// named here for the same reason hip.div names them.
func.func @mod_narrow_int(%ctx: !hip.context, %a: tensor<4xi8>,
                          %b: tensor<4xi8>, %init: tensor<4xi8>)
    -> tensor<4xi8> attributes {rock.kernel} {
  // expected-error @+2 {{hip.mod has no TOSA spelling for element type 'i8'}}
  // expected-error @+1 {{failed to legalize operation 'hip.mod'}}
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xi8>, tensor<4xi8>)
                     outs(%init : tensor<4xi8>) : tensor<4xi8>
  return %r : tensor<4xi8>
}

// -----

// Unsigned is the other width this pass cannot reach, since it has no type
// converter to rewrite signedness away.
func.func @mod_unsigned(%ctx: !hip.context, %a: tensor<4xui32>,
                        %b: tensor<4xui32>, %init: tensor<4xui32>)
    -> tensor<4xui32> attributes {rock.kernel} {
  // expected-error @+2 {{hip.mod has no TOSA spelling for element type 'ui32'}}
  // expected-error @+1 {{failed to legalize operation 'hip.mod'}}
  %r = hip.mod(%ctx) ins(%a, %b : tensor<4xui32>, tensor<4xui32>)
                     outs(%init : tensor<4xui32>) : tensor<4xui32>
  return %r : tensor<4xui32>
}

// -----

// ONNX Round is float-only, so an integer round cannot come from a valid model;
// the gate states the expansion's requirement rather than rejecting real input.
func.func @round_int(%ctx: !hip.context, %x: tensor<4xi32>,
                     %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+2 {{hip.round has no TOSA spelling for element type 'i32'}}
  // expected-error @+1 {{failed to legalize operation 'hip.round'}}
  %r = hip.round(%ctx) ins(%x : tensor<4xi32>)
                       outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

// f64 is the reachable one: ONNX has a double tensor type, so a double Round is
// a valid model, but TOSA has no f64 tensor type. Asking only for a FloatType
// would have built tosa.floor and tosa.ceil on an element type TOSA cannot
// represent, which verifies here and fails later. GemmConverter excludes f64
// for the same reason.
func.func @round_f64(%ctx: !hip.context, %x: tensor<4xf64>,
                     %init: tensor<4xf64>) -> tensor<4xf64>
    attributes {rock.kernel} {
  // expected-error @+2 {{hip.round has no TOSA spelling for element type 'f64'}}
  // expected-error @+1 {{failed to legalize operation 'hip.round'}}
  %r = hip.round(%ctx) ins(%x : tensor<4xf64>)
                       outs(%init : tensor<4xf64>) : tensor<4xf64>
  return %r : tensor<4xf64>
}

// -----

// bf16 rounds like the other two supported floats, with no f32 bridge.
// CHECK-LABEL: func.func @round_bf16
// CHECK: tosa.floor %arg1 : (tensor<4xbf16>) -> tensor<4xbf16>
// CHECK: tosa.ceil %arg1 : (tensor<4xbf16>) -> tensor<4xbf16>
// CHECK-NOT: hip.round
func.func @round_bf16(%ctx: !hip.context, %x: tensor<4xbf16>,
                      %init: tensor<4xbf16>) -> tensor<4xbf16>
    attributes {rock.kernel} {
  %r = hip.round(%ctx) ins(%x : tensor<4xbf16>)
                       outs(%init : tensor<4xbf16>) : tensor<4xbf16>
  return %r : tensor<4xbf16>
}

// -----

//===----------------------------------------------------------------------===//
// Left alone.
//===----------------------------------------------------------------------===//

// Only outlined kernels are rewritten; the host graph keeps its runtime ops.
// The float element type is the point: the rejection above follows from being
// inside a kernel, not from a judgement about the type, and the runtime's own
// fmod handles floats perfectly well.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: hip.mod
// CHECK: hip.round
// CHECK-NOT: tosa.intdiv
func.func @not_a_kernel(%ctx: !hip.context, %a: tensor<4xf32>,
                        %b: tensor<4xf32>, %init: tensor<4xf32>)
    -> tensor<4xf32> {
  %m = hip.mod(%ctx) ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
                     outs(%init : tensor<4xf32>) {fmod = 1 : i64} : tensor<4xf32>
  %r = hip.round(%ctx) ins(%m : tensor<4xf32>)
                       outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected shapes.
//===----------------------------------------------------------------------===//

// A dynamic shape gives the expansion no shape to build its constants at, and
// fails legalization rather than passing through.
func.func @round_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                               %init: tensor<?x8xf32>) -> tensor<?x8xf32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.round'}}
  %r = hip.round(%ctx) ins(%x : tensor<?x8xf32>)
                       outs(%init : tensor<?x8xf32>) : tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}

// -----

func.func @mod_dynamic_shape(%ctx: !hip.context, %a: tensor<?x8xi32>,
                             %b: tensor<?x8xi32>, %init: tensor<?x8xi32>)
    -> tensor<?x8xi32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.mod'}}
  %r = hip.mod(%ctx) ins(%a, %b : tensor<?x8xi32>, tensor<?x8xi32>)
                     outs(%init : tensor<?x8xi32>) : tensor<?x8xi32>
  return %r : tensor<?x8xi32>
}
