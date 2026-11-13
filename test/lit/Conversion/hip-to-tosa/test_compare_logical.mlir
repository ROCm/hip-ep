// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the comparison, logical and sign hip ops lower to TOSA inside a
// rock.kernel function, so rocMLIR can absorb them into a fused kernel, and
// verify the forms the conversion leaves alone or rejects.
//
// These ops are claimed by element type and shape rather than outright,
// because ONNX has no signless integers and ORT imports bool as ui8. An op
// this pass cannot express therefore stays a hip op for the runtime lowering
// instead of failing the pass -- see the "Forms this pass does not claim"
// section, which is the whole of the negative coverage here.
//
// There are deliberately no expected-error cases: every legality predicate
// checks exactly what its pattern accepts, down to operand element types and
// broadcastability, so no form reaches a pattern that then refuses it. That
// matters because an op marked illegal and not rewritten aborts the whole
// function's conversion, taking the fusible ops around it down too. A new
// diagnostic appearing in this file means a predicate and its pattern have
// drifted apart.
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
// pass run. Each declined form then gets its own chunk, so that a regression
// turning one into a pass failure cannot mask the cases after it.
//
// This test validates:
// - hip.equal maps to tosa.equal with its operands in order
// - hip.less maps to tosa.greater with its operands swapped
// - hip.and / hip.or / hip.not use the tosa.bitwise_* ops, which are the forms
//   rocMLIR can lower, and which agree with the logical ones on i1
// - hip.sign expands to select(x > 0, 1, select(0 > x, -1, 0))
// - The i1 result of a comparison does not block broadcasting of its operands
// - The hip context and the DPS outs operand are both dropped
// - The pass is a no-op on functions without rock.kernel
// - Dynamic shapes, un-broadcastable operands, mismatched operand element
//   types, non-i1 operands to the logical ops, ui8 booleans, unsigned
//   comparison operands, and the float and integer widths outside the TOSA
//   allow-lists (f64, f8, i4) all stay hip ops without failing the pass
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
// CHECK: tosa.bitwise_and %arg1, %arg2 : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.and
func.func @and(%ctx: !hip.context, %x: tensor<2x8xi1>, %y: tensor<2x8xi1>,
               %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xi1>, tensor<2x8xi1>)
                     outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// CHECK-LABEL: func.func @or
// CHECK: tosa.bitwise_or %arg1, %arg2 : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.or
func.func @or(%ctx: !hip.context, %x: tensor<2x8xi1>, %y: tensor<2x8xi1>,
              %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.or(%ctx) ins(%x, %y : tensor<2x8xi1>, tensor<2x8xi1>)
                    outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// `!x` is `x ^ true`, because neither tosa.logical_not nor tosa.bitwise_not
// has a rocMLIR lowering while tosa.bitwise_xor does.
// CHECK-LABEL: func.func @not
// CHECK: %[[ONES:.*]] = "tosa.const"() <{values = dense<true> : tensor<2x8xi1>}>
// CHECK: tosa.bitwise_xor %arg1, %[[ONES]] : (tensor<2x8xi1>, tensor<2x8xi1>) -> tensor<2x8xi1>
// CHECK-NOT: hip.not
// CHECK-NOT: tosa.logical_not
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
// CHECK: %[[SIGNUM:.*]] = tosa.select %[[POS]], %[[ONE]], %[[INNER]]
// Both comparisons above are false for NaN, so the selects alone would return
// zero. ONNX defines sign(NaN) = NaN and lib/Runtime/real/sign.cpp propagates
// it, so an ordered self-compare forwards the input in that one case and keeps
// a fused model agreeing with the unfused one.
// CHECK: %[[ORDERED:.*]] = tosa.equal %arg1, %arg1
// CHECK: tosa.select %[[ORDERED]], %[[SIGNUM]], %arg1
// CHECK-NOT: hip.sign
func.func @sign(%ctx: !hip.context, %x: tensor<2x8xf32>,
                %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<2x8xf32>)
                      outs(%init : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// ONNX Sign is defined over signed integers too, where the constants are
// integer rather than float splats. Integers have no NaN, so the self-compare
// guard is float-only and must not appear here.
// CHECK-LABEL: func.func @sign_i32
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<4xi32>}>
// CHECK: %[[POS:.*]] = tosa.greater %arg1, %[[ZERO]]
// CHECK: %[[ISNEG:.*]] = tosa.greater %[[ZERO]], %arg1
// CHECK: %[[NEG:.*]] = "tosa.const"() <{values = dense<-1> : tensor<4xi32>}>
// CHECK: %[[INNER:.*]] = tosa.select %[[ISNEG]], %[[NEG]], %[[ZERO]]
// CHECK: %[[ONE:.*]] = "tosa.const"() <{values = dense<1> : tensor<4xi32>}>
// CHECK: tosa.select %[[POS]], %[[ONE]], %[[INNER]]
// CHECK-NOT: tosa.equal
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
// Forms this pass does not claim. These ops are legal by element type rather
// than outright, so an unsupported one is left as a hip op and reaches the
// runtime lowering that handles it; the pass still succeeds. That is the point
// of the gates -- failing here would take the fusible ops in the same function
// down with it.
//
// These all share one chunk since none of them produce a diagnostic.
//===----------------------------------------------------------------------===//

// ORT imports ONNX bool as ui8, not i1 (see the ui8 cases in
// test/lit/Conversion/hip-to-llvm/test_and.mlir). TOSA has no unsigned
// integers, so the whole ui8 boolean family stays on the runtime path.
// CHECK-LABEL: func.func @and_ui8_bool
// CHECK: hip.and
// CHECK-NOT: tosa.bitwise_and
func.func @and_ui8_bool(%ctx: !hip.context, %x: tensor<2x8xui8>,
                        %y: tensor<2x8xui8>, %init: tensor<2x8xui8>)
    -> tensor<2x8xui8> attributes {rock.kernel} {
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xui8>, tensor<2x8xui8>)
                     outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// CHECK-LABEL: func.func @not_ui8_bool
// CHECK: hip.not
// CHECK-NOT: tosa.bitwise_xor
func.func @not_ui8_bool(%ctx: !hip.context, %x: tensor<2x8xui8>,
                        %init: tensor<2x8xui8>) -> tensor<2x8xui8>
    attributes {rock.kernel} {
  %r = hip.not(%ctx) ins(%x : tensor<2x8xui8>)
                     outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// OnnxToHip preserves the ONNX result type, so a lowered onnx.Greater can
// arrive as a hip.less returning ui8 -- exactly the shape of
// test/lit/Conversion/onnx-to-hip/test_greater.mlir's greater_6d_scalar.
// CHECK-LABEL: func.func @less_ui8_result
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_ui8_result(%ctx: !hip.context, %x: tensor<2x8xf32>,
                           %y: tensor<2x8xf32>, %init: tensor<2x8xui8>)
    -> tensor<2x8xui8> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                      outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// ONNX Equal/Less also accept unsigned operands. TOSA integers are signless,
// so comparing them as TOSA would read a ui8 255 as -1.
// CHECK-LABEL: func.func @equal_unsigned_operands
// CHECK: hip.equal
// CHECK-NOT: tosa.equal
func.func @equal_unsigned_operands(%ctx: !hip.context, %x: tensor<2x8xui8>,
                                   %y: tensor<2x8xui8>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xui8>, tensor<2x8xui8>)
                       outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// Bitwise and logical agree only on i1, so a wider integer keeps its hip op
// rather than being given bitwise semantics it does not ask for.
// CHECK-LABEL: func.func @and_non_i1
// CHECK: hip.and
// CHECK-NOT: tosa.bitwise_and
func.func @and_non_i1(%ctx: !hip.context, %x: tensor<2x8xi8>,
                      %y: tensor<2x8xi8>, %init: tensor<2x8xi8>)
    -> tensor<2x8xi8> attributes {rock.kernel} {
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xi8>, tensor<2x8xi8>)
                     outs(%init : tensor<2x8xi8>) : tensor<2x8xi8>
  return %r : tensor<2x8xi8>
}

// Dynamic shapes give the pattern no static shape to reason about.
// CHECK-LABEL: func.func @less_dynamic_shape
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                              %y: tensor<?x8xf32>, %init: tensor<?x8xi1>)
    -> tensor<?x8xi1> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<?x8xf32>, tensor<?x8xf32>)
                      outs(%init : tensor<?x8xi1>) : tensor<?x8xi1>
  return %r : tensor<?x8xi1>
}

// Both TOSA comparisons carry SameOperandsElementType, so mismatched operands
// have no spelling even though the result is i1 either way.
// CHECK-LABEL: func.func @equal_element_type_mismatch
// CHECK: hip.equal
// CHECK-NOT: tosa.equal
func.func @equal_element_type_mismatch(%ctx: !hip.context,
                                       %x: tensor<2x8xf32>,
                                       %y: tensor<2x8xf16>,
                                       %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.equal(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// TOSA's float tensor constraint is AnyFloat, so f64 would pass the verifier
// and then have no lowering.
// CHECK-LABEL: func.func @less_f64
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_f64(%ctx: !hip.context, %x: tensor<2x8xf64>,
                    %y: tensor<2x8xf64>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf64>, tensor<2x8xf64>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// The same for the sign expansion, whose greater/select would be built at f64.
// CHECK-LABEL: func.func @sign_f64
// CHECK: hip.sign
// CHECK-NOT: tosa.select
func.func @sign_f64(%ctx: !hip.context, %x: tensor<2x8xf64>,
                    %init: tensor<2x8xf64>) -> tensor<2x8xf64>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<2x8xf64>)
                      outs(%init : tensor<2x8xf64>) : tensor<2x8xf64>
  return %r : tensor<2x8xf64>
}

// i1 cannot represent the -1 that Sign needs for negative inputs.
// CHECK-LABEL: func.func @sign_i1
// CHECK: hip.sign
// CHECK-NOT: tosa.select
func.func @sign_i1(%ctx: !hip.context, %x: tensor<2x8xi1>,
                   %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<2x8xi1>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// Rank equalization prepends 1s, so tensor<4xf32> becomes 1x4. The trailing 4
// still cannot broadcast to 8. The legality predicate runs the same broadcast
// check the pattern does, so this is declined up front rather than claimed and
// then failed.
// CHECK-LABEL: func.func @less_incompatible_broadcast
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_incompatible_broadcast(%ctx: !hip.context,
                                       %x: tensor<2x8xf32>, %y: tensor<4xf32>,
                                       %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<4xf32>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// Same check on the logical ops, whose operands the predicate has to inspect
// separately: the result here is a perfectly good static i1 tensor, and only
// the operand is un-broadcastable.
// CHECK-LABEL: func.func @and_unbroadcastable_operand
// CHECK: hip.and
// CHECK-NOT: tosa.bitwise_and
func.func @and_unbroadcastable_operand(%ctx: !hip.context, %x: tensor<2x8xi1>,
                                       %y: tensor<4xi1>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.and(%ctx) ins(%x, %y : tensor<2x8xi1>, tensor<4xi1>)
                     outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// A static i1 result does not imply a static operand, so the predicate checks
// the operand too.
// CHECK-LABEL: func.func @not_dynamic_operand
// CHECK: hip.not
// CHECK-NOT: tosa.bitwise_xor
func.func @not_dynamic_operand(%ctx: !hip.context, %x: tensor<?x8xi1>,
                               %init: tensor<2x8xi1>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %r = hip.not(%ctx) ins(%x : tensor<?x8xi1>)
                     outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// SignConverter builds every constant and comparison at the result type, so it
// needs the input to match exactly; a static result with a dynamic input is
// not enough.
// CHECK-LABEL: func.func @sign_mismatched_input
// CHECK: hip.sign
// CHECK-NOT: tosa.select
func.func @sign_mismatched_input(%ctx: !hip.context, %x: tensor<?x8xf32>,
                                 %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<?x8xf32>)
                      outs(%init : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

// The float gate is an allow-list of f16/bf16/f32, not "any float but f64":
// Tosa_FloatTensor is AnyFloat, so an f8 tensor would satisfy the verifiers
// while nothing downstream can lower it.
// CHECK-LABEL: func.func @less_f8
// CHECK: hip.less
// CHECK-NOT: tosa.greater
func.func @less_f8(%ctx: !hip.context, %x: tensor<2x8xf8E4M3FN>,
                   %y: tensor<2x8xf8E4M3FN>, %init: tensor<2x8xi1>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %r = hip.less(%ctx) ins(%x, %y : tensor<2x8xf8E4M3FN>, tensor<2x8xf8E4M3FN>)
                      outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// -----

// Likewise the integer gate names the widths ONNX produces rather than taking
// any signless integer, since Tosa_Int would also admit i4 and i128.
// CHECK-LABEL: func.func @sign_i4
// CHECK: hip.sign
// CHECK-NOT: tosa.select
func.func @sign_i4(%ctx: !hip.context, %x: tensor<4xi4>, %init: tensor<4xi4>)
    -> tensor<4xi4> attributes {rock.kernel} {
  %r = hip.sign(%ctx) ins(%x : tensor<4xi4>)
                      outs(%init : tensor<4xi4>) : tensor<4xi4>
  return %r : tensor<4xi4>
}
