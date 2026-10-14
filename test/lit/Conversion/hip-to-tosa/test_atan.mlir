// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that hip.atan, the lowering of ONNX Atan, becomes TOSA inside a
// rock.kernel function. atan is already in FuseROCMlir's pointwise list, so an
// atan sitting behind a matmul is outlined into the kernel whether or not this
// pass can lower it; left unconverted it reaches rocMLIR as a hip op that
// rocMLIR cannot compile.
//
// Unlike round and mod, this expansion approximates. TOSA has no arctangent
// and no identity reaches one from the transcendentals it does carry, so the
// expansion is the Cephes atanf algorithm: fold the domain onto
// [0, tan(pi/8)] at the two points tan(pi/8) and tan(3pi/8), then finish with
// an odd minimax polynomial. The runtime calls the device atanf, so the two
// agree to within their error bounds rather than bit-for-bit.
//
// COVERAGE:
// - The full expansion: abs, the two fold compares, both reduced arguments,
//   the offset select, the degree-3 Horner polynomial in z = r*r
// - Both reduced arguments are computed on every lane, since TOSA has no
//   control flow; the unselected one can be an infinity and tosa.select
//   discards it without doing arithmetic on it
// - The sign goes back on with a select rather than a copysign, and the signed
//   zero is restored separately because abs erases it and `0 > x` is false
//   for -0
// - f16 and bf16 are bridged through f32, as the atan runtime's f16 kernel is
//   __half2float / atanf / __float2half
// - f32 needs no bridge, so no tosa.cast appears
// - f64 is named, which ONNX allows and TOSA has no tensor type for, and so
//   are integers, which ONNX Atan never produces
// - The expansion emits no tosa.logical_* op, which rocMLIR's
//   RockTosaToElementwise has no pattern for
// - atan converts behind a matmul anchor, the shape hip-fuse-rocmlir makes
// - Outside a rock.kernel the op is left alone for the runtime
// - Dynamic shapes are rejected, matching the sibling elementwise ops
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// The full expansion. %[[AX]] is |x|; %[[HI]] and %[[MID]] pick the fold, and
// %[[R]] is the argument folded onto [0, tan(pi/8)]. The reciprocal in
// %[[HIARG]] is an infinity at x = 0, but that lane is never selected, and at
// x = inf it is the 0 that makes the result exactly pi/2.
// CHECK-LABEL: func.func @atan_f32
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<4xf32>}>
// CHECK-DAG: %[[ONE:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4xf32>}>
// CHECK-DAG: %[[MONE:.*]] = "tosa.const"() <{values = dense<-1.000000e+00> : tensor<4xf32>}>
// CHECK: %[[AX:.*]] = tosa.abs %arg1
// CHECK: %[[HI:.*]] = tosa.greater %[[AX]], %{{.*}}
// CHECK: %[[MID:.*]] = tosa.greater %[[AX]], %{{.*}}
// CHECK: %[[INV:.*]] = tosa.reciprocal %[[AX]]
// CHECK: %[[HIARG:.*]] = tosa.mul %[[INV]], %[[MONE]]
// CHECK: %[[NUM:.*]] = tosa.sub %[[AX]], %[[ONE]]
// CHECK: %[[DEN:.*]] = tosa.add %[[AX]], %[[ONE]]
// CHECK: %[[INVDEN:.*]] = tosa.reciprocal %[[DEN]]
// CHECK: %[[MIDARG:.*]] = tosa.mul %[[NUM]], %[[INVDEN]]
// CHECK: %[[LOWMID:.*]] = tosa.select %[[MID]], %[[MIDARG]], %[[AX]]
// CHECK: %[[R:.*]] = tosa.select %[[HI]], %[[HIARG]], %[[LOWMID]]
// CHECK: %[[MIDOFF:.*]] = tosa.select %[[MID]], %{{.*}}, %[[ZERO]]
// CHECK: %[[OFF:.*]] = tosa.select %[[HI]], %{{.*}}, %[[MIDOFF]]
// CHECK: %[[Z:.*]] = tosa.mul %[[R]], %[[R]]
// The three Horner steps, then r*z*poly + r folded onto the offset.
// CHECK-COUNT-3: tosa.add %{{.*}}, %{{.*}}
// CHECK: %[[NEG:.*]] = tosa.greater %[[ZERO]], %arg1
// CHECK: %[[FLIP:.*]] = tosa.mul %{{.*}}, %[[MONE]]
// CHECK: %[[SIGNED:.*]] = tosa.select %[[NEG]], %[[FLIP]], %{{.*}}
// atan(+-0) is +-0, so the input is handed straight back at both zeros.
// CHECK: %[[ISZERO:.*]] = tosa.equal %arg1, %[[ZERO]]
// CHECK: tosa.select %[[ISZERO]], %arg1, %[[SIGNED]]
// f32 needs no widening, so nothing casts.
// CHECK-NOT: tosa.cast
// CHECK-NOT: hip.atan
func.func @atan_f32(%ctx: !hip.context, %x: tensor<4xf32>,
                    %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.atan(%ctx) ins(%x : tensor<4xf32>)
                      outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// f16 is bridged through f32 rather than evaluated in place. The runtime's f16
// atan kernel is __half2float / atanf / __float2half, and a degree-9 polynomial
// in 11 mantissa bits would throw away most of what the minimax fit buys.
// CHECK-LABEL: func.func @atan_f16
// CHECK: %[[UP:.*]] = tosa.cast %arg1 : (tensor<2x8xf16>) -> tensor<2x8xf32>
// CHECK: tosa.abs %[[UP]] : (tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK: tosa.cast %{{.*}} : (tensor<2x8xf32>) -> tensor<2x8xf16>
// CHECK-NOT: hip.atan
func.func @atan_f16(%ctx: !hip.context, %x: tensor<2x8xf16>,
                    %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.atan(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

// bf16 takes the same bridge, and for the stronger reason: it carries 8
// mantissa bits.
// CHECK-LABEL: func.func @atan_bf16
// CHECK: %[[UP:.*]] = tosa.cast %arg1 : (tensor<4xbf16>) -> tensor<4xf32>
// CHECK: tosa.abs %[[UP]] : (tensor<4xf32>) -> tensor<4xf32>
// CHECK: tosa.cast %{{.*}} : (tensor<4xf32>) -> tensor<4xbf16>
// CHECK-NOT: hip.atan
func.func @atan_bf16(%ctx: !hip.context, %x: tensor<4xbf16>,
                     %init: tensor<4xbf16>) -> tensor<4xbf16>
    attributes {rock.kernel} {
  %r = hip.atan(%ctx) ins(%x : tensor<4xbf16>)
                      outs(%init : tensor<4xbf16>) : tensor<4xbf16>
  return %r : tensor<4xbf16>
}

// -----

// The expansion combines nothing, so it emits no tosa.logical_* op to begin
// with. rocMLIR's RockTosaToElementwise has a pattern for none of them while
// marking every surviving tosa op illegal, which is why round and mod spell
// their predicates with the bitwise ops; atan has no predicate to combine.
// CHECK-LABEL: func.func @atan_no_logical_ops
// CHECK-NOT: tosa.logical_and
// CHECK-NOT: tosa.logical_or
// CHECK-NOT: tosa.logical_not
// CHECK-NOT: tosa.logical_xor
// CHECK-NOT: hip.atan
func.func @atan_no_logical_ops(%ctx: !hip.context, %x: tensor<4xf32>,
                               %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.atan(%ctx) ins(%x : tensor<4xf32>)
                      outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

// The shape hip-fuse-rocmlir produces: a matmul anchor with the pointwise op
// behind it and a ub.poison standing in for the context. atan is in the
// pointwise list, so an atan after a matmul is outlined into the kernel
// whether or not it can be lowered; left unconverted it reaches rocMLIR as a
// hip op fed by a ub.poison context, neither of which rocMLIR knows.
// CHECK-LABEL: func.func @rocMlir0
// CHECK: ub.poison : !hip.context
// CHECK: %[[MM:.*]] = tosa.matmul
// CHECK: tosa.abs
// CHECK: tosa.reciprocal
// CHECK: tosa.select
// CHECK-NOT: hip.atan
// CHECK-NOT: hip.matmul
func.func @rocMlir0(%a: tensor<8x64x64xf32>, %w: tensor<64x64xf32>)
    -> tensor<8x64x64xf32> attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %m_init = tensor.empty() : tensor<8x64x64xf32>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<8x64x64xf32>, tensor<64x64xf32>)
                        outs(%m_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  %r_init = tensor.empty() : tensor<8x64x64xf32>
  %r = hip.atan(%ctx) ins(%m : tensor<8x64x64xf32>)
                      outs(%r_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  return %r : tensor<8x64x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Element types with no TOSA spelling.
//===----------------------------------------------------------------------===//

// f64 is the reachable one: ONNX has a double tensor type, so a double Atan is
// a valid model, but TOSA has no f64 tensor type. RoundConverter and
// GemmConverter exclude it for the same reason.
func.func @atan_f64(%ctx: !hip.context, %x: tensor<4xf64>,
                    %init: tensor<4xf64>) -> tensor<4xf64>
    attributes {rock.kernel} {
  // expected-error @+2 {{hip.atan has no TOSA spelling for element type 'f64'}}
  // expected-error @+1 {{failed to legalize operation 'hip.atan'}}
  %r = hip.atan(%ctx) ins(%x : tensor<4xf64>)
                      outs(%init : tensor<4xf64>) : tensor<4xf64>
  return %r : tensor<4xf64>
}

// -----

// ONNX Atan is float-only, so an integer atan cannot come from a valid model;
// the gate states the expansion's requirement rather than rejecting real input.
func.func @atan_int(%ctx: !hip.context, %x: tensor<4xi32>,
                    %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+2 {{hip.atan has no TOSA spelling for element type 'i32'}}
  // expected-error @+1 {{failed to legalize operation 'hip.atan'}}
  %r = hip.atan(%ctx) ins(%x : tensor<4xi32>)
                      outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// Outside a kernel, and dynamic shapes.
//===----------------------------------------------------------------------===//

// The pass runs only inside a rock.kernel. Everywhere else hip.atan is left for
// the runtime's own atan, which handles this perfectly well -- the rejections
// above follow from being inside a kernel, not from a judgement about the op.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: hip.atan
// CHECK-NOT: tosa.abs
func.func @not_a_kernel(%ctx: !hip.context, %x: tensor<4xf32>,
                        %init: tensor<4xf32>) -> tensor<4xf32> {
  %r = hip.atan(%ctx) ins(%x : tensor<4xf32>)
                      outs(%init : tensor<4xf32>) : tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @atan_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                              %init: tensor<?x8xf32>) -> tensor<?x8xf32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.atan'}}
  %r = hip.atan(%ctx) ins(%x : tensor<?x8xf32>)
                      outs(%init : tensor<?x8xf32>) : tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}
