// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that hip.div, the lowering of ONNX Div, becomes TOSA inside a
// rock.kernel function, so a divide sitting behind a matmul or convolution is
// absorbed into the fused kernel instead of stranding a hip op that rocMLIR
// cannot read.
//
// TOSA has no single divide, so the conversion splits on element type the way
// MIGraphXToTosa splits migraphx.div: integers become tosa.intdiv, and floats
// become a tosa.reciprocal followed by a tosa.mul, which is what tosa.intdiv's
// own description prescribes. That split is also why hip.div cannot join the
// BinaryConverter template its add/sub/min/max/mul siblings share.
//
// COVERAGE:
// - Floats become reciprocal-then-multiply, with the zero shift tosa.mul's
//   verifier requires
// - Signless i32 and i64, the only widths tosa.intdiv accepts, become intdiv
// - The reciprocal is taken at the divisor's own shape, so a broadcasting
//   divisor is reciprocated once per distinct element rather than once per
//   result element
// - A lower-rank divisor is reshaped with leading 1s first, since TOSA
//   broadcasts size-1 dimensions only once both operands carry the result rank
// - A divide behind a matmul anchor converts alongside it
// - Element types with no TOSA spelling -- unsigned, and integer widths
//   narrower than 32 -- are rejected with the type named, since a hip.div left
//   inside a kernel is not something rocMLIR can compile either
// - The pass is a no-op on functions without rock.kernel, so a host-graph
//   divide keeps its runtime op whatever its element type
// - Dynamic shapes are rejected, matching the hip.add sibling
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Element types that convert.
//===----------------------------------------------------------------------===//

// The float spelling: reciprocate the divisor, then multiply. The shift is the
// same zero tensor<1xi8> the other tosa.mul users build, required here because
// tosa.mul's verifier rejects a non-zero shift on float operands.
// CHECK-LABEL: func.func @div_f32
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %arg2 : (tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: tosa.mul %arg1, %[[RECIP]], %[[SHIFT]] : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<1xi8>) -> tensor<2x8xf32>
// CHECK-NOT: hip.div
func.func @div_f32(%ctx: !hip.context, %x: tensor<2x8xf32>, %y: tensor<2x8xf32>,
                   %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %r = hip.div(%ctx) ins(%x, %y : tensor<2x8xf32>, tensor<2x8xf32>)
                     outs(%init : tensor<2x8xf32>) : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// -----

// Integers take tosa.intdiv instead, which truncates towards zero the way ONNX
// Div does on integer input.
// CHECK-LABEL: func.func @div_i32
// CHECK: tosa.intdiv %arg1, %arg2 : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
// CHECK-NOT: tosa.reciprocal
// CHECK-NOT: hip.div
func.func @div_i32(%ctx: !hip.context, %x: tensor<2x8xi32>, %y: tensor<2x8xi32>,
                   %init: tensor<2x8xi32>) -> tensor<2x8xi32>
    attributes {rock.kernel} {
  %r = hip.div(%ctx) ins(%x, %y : tensor<2x8xi32>, tensor<2x8xi32>)
                     outs(%init : tensor<2x8xi32>) : tensor<2x8xi32>
  return %r : tensor<2x8xi32>
}

// -----

// i64 is the other half of tosa.intdiv's Tosa_Int32Or64Tensor operand type.
// CHECK-LABEL: func.func @div_i64
// CHECK: tosa.intdiv %arg1, %arg2 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
// CHECK-NOT: hip.div
func.func @div_i64(%ctx: !hip.context, %x: tensor<4xi64>, %y: tensor<4xi64>,
                   %init: tensor<4xi64>) -> tensor<4xi64>
    attributes {rock.kernel} {
  %r = hip.div(%ctx) ins(%x, %y : tensor<4xi64>, tensor<4xi64>)
                     outs(%init : tensor<4xi64>) : tensor<4xi64>
  return %r : tensor<4xi64>
}

// -----

//===----------------------------------------------------------------------===//
// Broadcasting divisors.
//===----------------------------------------------------------------------===//

// The reciprocal stays at tensor<4x1xf16>, the divisor's own shape, rather than
// being widened to the result first: four reciprocals instead of thirty-two,
// with the multiply broadcasting them back up. Taking it at the result shape
// would verify just as well, which is why this is pinned.
// CHECK-LABEL: func.func @div_broadcast_divisor
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %arg2 : (tensor<4x1xf16>) -> tensor<4x1xf16>
// CHECK: tosa.mul %arg1, %[[RECIP]], %{{.*}} : (tensor<4x8xf16>, tensor<4x1xf16>, tensor<1xi8>) -> tensor<4x8xf16>
// CHECK-NOT: hip.div
func.func @div_broadcast_divisor(%ctx: !hip.context, %x: tensor<4x8xf16>,
                                 %y: tensor<4x1xf16>, %init: tensor<4x8xf16>)
    -> tensor<4x8xf16> attributes {rock.kernel} {
  %r = hip.div(%ctx) ins(%x, %y : tensor<4x8xf16>, tensor<4x1xf16>)
                     outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return %r : tensor<4x8xf16>
}

// -----

// Dividing by a scalar is the common shape -- an attention score divided by
// sqrt(head_dim) arrives exactly like this. hip broadcasting rank-extends the
// way ONNX does, so the rank-0 divisor is reshaped to 1x1x1x1 before TOSA can
// broadcast it, and the whole division collapses to a single reciprocal.
// CHECK-LABEL: func.func @div_rank_extending
// CHECK: %[[SHAPE:.*]] = tosa.const_shape {values = dense<1> : tensor<4xindex>}
// CHECK: %[[RESHAPED:.*]] = tosa.reshape %arg2, %[[SHAPE]] : (tensor<f16>, !tosa.shape<4>) -> tensor<1x1x1x1xf16>
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %[[RESHAPED]] : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
// CHECK: tosa.mul %arg1, %[[RECIP]], %{{.*}} : (tensor<1x64x112x112xf16>, tensor<1x1x1x1xf16>, tensor<1xi8>) -> tensor<1x64x112x112xf16>
// CHECK-NOT: hip.div
func.func @div_rank_extending(%ctx: !hip.context, %x: tensor<1x64x112x112xf16>,
                              %y: tensor<f16>, %init: tensor<1x64x112x112xf16>)
    -> tensor<1x64x112x112xf16> attributes {rock.kernel} {
  %r = hip.div(%ctx) ins(%x, %y : tensor<1x64x112x112xf16>, tensor<f16>)
                     outs(%init : tensor<1x64x112x112xf16>)
                     : tensor<1x64x112x112xf16>
  return %r : tensor<1x64x112x112xf16>
}

// -----

//===----------------------------------------------------------------------===//
// A divide behind an anchor, in the shape hip-fuse-rocmlir produces.
//===----------------------------------------------------------------------===//

// The reason for converting hip.div at all, and what separates it from the
// shape ops: hip.div is already in FuseROCMlir's pointwise list, so a divide
// after a matmul is outlined into the kernel whether or not it can be lowered.
// Left unconverted it reaches rocMLIR as a hip op fed by a ub.poison context,
// neither of which rocMLIR knows. The ub.poison and the tensor.empty buffers
// this pass does not claim survive untouched for the canonicalizer.
// CHECK-LABEL: func.func @rocMlir0
// CHECK: ub.poison : !hip.context
// CHECK: tensor.empty() : tensor<8x64x64xf32>
// CHECK: %[[MM:.*]] = tosa.matmul
// CHECK: %[[OUT:.*]] = tosa.reshape %[[MM]], %{{.*}} -> tensor<8x64x64xf32>
// CHECK: %[[RECIP:.*]] = tosa.reciprocal %arg2 : (tensor<8x64x64xf32>) -> tensor<8x64x64xf32>
// CHECK: tosa.mul %[[OUT]], %[[RECIP]], %{{.*}} -> tensor<8x64x64xf32>
// CHECK-NOT: hip.div
// CHECK-NOT: hip.matmul
func.func @rocMlir0(%a: tensor<8x64x64xf32>, %w: tensor<64x64xf32>,
                    %d: tensor<8x64x64xf32>) -> tensor<8x64x64xf32>
    attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %m_init = tensor.empty() : tensor<8x64x64xf32>
  %m = hip.matmul(%ctx) ins(%a, %w : tensor<8x64x64xf32>, tensor<64x64xf32>)
                        outs(%m_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  %d_init = tensor.empty() : tensor<8x64x64xf32>
  %r = hip.div(%ctx) ins(%m, %d : tensor<8x64x64xf32>, tensor<8x64x64xf32>)
                     outs(%d_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  return %r : tensor<8x64x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Element types with no TOSA spelling.
//===----------------------------------------------------------------------===//

// tosa.intdiv takes signless i32 and i64 only, and nothing upstream narrows
// what hip.div can carry: the operand constraint is AnyRankedTensor and
// OnnxToHip copies the ONNX element type through verbatim, so an unsigned
// divide reaches this pass intact.
//
// Rejecting is the only option. This pass runs only inside a rock.kernel, and
// rocMLIR compiles that kernel to an ELF, so a hip.div left behind fails there
// anyway -- later, and reported as an op from an unknown dialect rather than as
// the unsupported element type it is. The diagnostic here names the type.
func.func @div_unsigned(%ctx: !hip.context, %x: tensor<4xui32>,
                        %y: tensor<4xui32>, %init: tensor<4xui32>)
    -> tensor<4xui32> attributes {rock.kernel} {
  // expected-error @+2 {{hip.div has no TOSA spelling for element type 'ui32'}}
  // expected-error @+1 {{failed to legalize operation 'hip.div'}}
  %r = hip.div(%ctx) ins(%x, %y : tensor<4xui32>, tensor<4xui32>)
                     outs(%init : tensor<4xui32>) : tensor<4xui32>
  return %r : tensor<4xui32>
}

// -----

// The same holds for the widths below 32 that the runtime can name but
// tosa.intdiv cannot take. MIGraphXToTosa does no better here: on a sub-32-bit
// signed divide it emits a tosa.intdiv that fails the verifier, rather than
// rejecting it.
func.func @div_narrow_int(%ctx: !hip.context, %x: tensor<4xi8>,
                          %y: tensor<4xi8>, %init: tensor<4xi8>)
    -> tensor<4xi8> attributes {rock.kernel} {
  // expected-error @+2 {{hip.div has no TOSA spelling for element type 'i8'}}
  // expected-error @+1 {{failed to legalize operation 'hip.div'}}
  %r = hip.div(%ctx) ins(%x, %y : tensor<4xi8>, tensor<4xi8>)
                     outs(%init : tensor<4xi8>) : tensor<4xi8>
  return %r : tensor<4xi8>
}

// -----

// Only outlined kernels are rewritten; the host graph keeps its runtime ops.
// The unsigned element type is the point: rejection above is a consequence of
// being inside a kernel, not a judgement about the type, and the runtime names
// ui32 perfectly well. Here the same divide is left alone.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: hip.div
// CHECK-NOT: tosa.reciprocal
func.func @not_a_kernel(%ctx: !hip.context, %x: tensor<4xui32>,
                        %y: tensor<4xui32>, %init: tensor<4xui32>)
    -> tensor<4xui32> {
  %r = hip.div(%ctx) ins(%x, %y : tensor<4xui32>, tensor<4xui32>)
                     outs(%init : tensor<4xui32>) : tensor<4xui32>
  return %r : tensor<4xui32>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected form.
//===----------------------------------------------------------------------===//

// Only the element type makes hip.div conditionally legal. Once the type is one
// TOSA can express, the shape rules are the sibling binary ops': a dynamic
// shape gives the pattern nothing to reason about and fails legalization rather
// than passing through.
func.func @div_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf32>,
                             %y: tensor<?x8xf32>, %init: tensor<?x8xf32>)
    -> tensor<?x8xf32> attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.div'}}
  %r = hip.div(%ctx) ins(%x, %y : tensor<?x8xf32>, tensor<?x8xf32>)
                     outs(%init : tensor<?x8xf32>) : tensor<?x8xf32>
  return %r : tensor<?x8xf32>
}
