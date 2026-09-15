// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify that hip.expand, the lowering of ONNX Expand, becomes the broadcast
// form rocMLIR understands, so the broadcast is folded into the neighbouring
// kernel instead of costing a wrap_expand kernel that materialises the result.
//
// TOSA has no broadcast op. rocMLIR spells one as a multiply against a tensor
// of ones at the result shape, relying on the implicit broadcasting of TOSA's
// elementwise ops, and MIGraphXToTosa lowers migraphx.multibroadcast -- the op
// ONNX Expand maps to on that path -- exactly this way. The multiply does not
// survive: TosaToRock's mulBroadcast matches the idiom, recognises the ones
// through isConstantOne, drops the multiply and rewrites the broadcast into a
// rock.transform, which is a coordinate remap rather than a copy.
//
// COVERAGE:
// - A same-rank broadcast becomes a bare tosa.mul against splat ones
// - A rank-extending broadcast prepends 1s with tosa.reshape first, since TOSA
//   broadcasts size-1 dimensions only once both operands carry the result rank
// - Integer and rank-0 inputs produce the same shape of output
// - The ones constant lands on the second operand and the shift is zero, which
//   is the form mulBroadcast and tosa.mul's own verifier require
// - A dynamically shaped expand is left alone for the runtime rather than
//   failing the pass: OnnxToHip emits that form deliberately, reading the
//   extents back from the device
// - The pass is a no-op on functions without rock.kernel
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Broadcasts that convert.
//===----------------------------------------------------------------------===//

// A leading-dimension broadcast, which needs no rank fixing. This is the shape
// rocMLIR's own mixr-to-tosa-ops.mlir pins for migraphx.multibroadcast, down to
// the f16 element type and the splat ones constant.
// CHECK-LABEL: func.func @expand_leading_dim
// CHECK-DAG: %[[ONES:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<64x768x2304xf16>}>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: tosa.mul %arg1, %[[ONES]], %[[SHIFT]] : (tensor<1x768x2304xf16>, tensor<64x768x2304xf16>, tensor<1xi8>) -> tensor<64x768x2304xf16>
// CHECK-NOT: hip.expand
func.func @expand_leading_dim(%ctx: !hip.context, %x: tensor<1x768x2304xf16>,
                              %s: tensor<3xi64>, %init: tensor<64x768x2304xf16>)
    -> tensor<64x768x2304xf16> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<1x768x2304xf16>, tensor<3xi64>)
                        outs(%init : tensor<64x768x2304xf16>)
                        : tensor<64x768x2304xf16>
  return %r : tensor<64x768x2304xf16>
}

// -----

// ONNX Expand rank-extends the way NumPy does, so a rank-2 input broadcast to a
// rank-3 result is right-aligned and padded with a leading 1 before the
// multiply. Both remaining dimensions broadcast: 3 passes through, 1 stretches
// to 6.
// CHECK-LABEL: func.func @expand_rank_extending
// CHECK-DAG: %[[ONES:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<2x3x6xf32>}>
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[1, 3, 1]> : tensor<3xindex>}
// CHECK: %[[RESHAPED:.*]] = tosa.reshape %arg1, %[[SHAPE]] : (tensor<3x1xf32>, !tosa.shape<3>) -> tensor<1x3x1xf32>
// CHECK: tosa.mul %[[RESHAPED]], %[[ONES]], %{{.*}} : (tensor<1x3x1xf32>, tensor<2x3x6xf32>, tensor<1xi8>) -> tensor<2x3x6xf32>
// CHECK-NOT: hip.expand
func.func @expand_rank_extending(%ctx: !hip.context, %x: tensor<3x1xf32>,
                                 %s: tensor<3xi64>, %init: tensor<2x3x6xf32>)
    -> tensor<2x3x6xf32> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<3x1xf32>, tensor<3xi64>)
                        outs(%init : tensor<2x3x6xf32>) : tensor<2x3x6xf32>
  return %r : tensor<2x3x6xf32>
}

// -----

// A rank-0 source is the widest rank extension there is; every dimension of the
// result comes from the broadcast.
// CHECK-LABEL: func.func @expand_scalar_source
// CHECK-DAG: %[[ONES:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4x8xf32>}>
// CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<1> : tensor<2xindex>}
// CHECK: %[[RESHAPED:.*]] = tosa.reshape %arg1, %[[SHAPE]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK: tosa.mul %[[RESHAPED]], %[[ONES]], %{{.*}} -> tensor<4x8xf32>
// CHECK-NOT: hip.expand
func.func @expand_scalar_source(%ctx: !hip.context, %x: tensor<f32>,
                                %s: tensor<2xi64>, %init: tensor<4x8xf32>)
    -> tensor<4x8xf32> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<f32>, tensor<2xi64>)
                        outs(%init : tensor<4x8xf32>) : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

// The ones constant follows the element type, and the integer case is why the
// shift operand has to be zero: tosa.mul right-shifts the product of integer
// inputs, so any other shift would scale the result rather than copy it.
// CHECK-LABEL: func.func @expand_integer
// CHECK-DAG: %[[ONES:.*]] = "tosa.const"() <{values = dense<1> : tensor<3x5xi32>}>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}>
// CHECK: tosa.mul %arg1, %[[ONES]], %[[SHIFT]] : (tensor<1x5xi32>, tensor<3x5xi32>, tensor<1xi8>) -> tensor<3x5xi32>
// CHECK-NOT: hip.expand
func.func @expand_integer(%ctx: !hip.context, %x: tensor<1x5xi32>,
                          %s: tensor<2xi64>, %init: tensor<3x5xi32>)
    -> tensor<3x5xi32> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<1x5xi32>, tensor<2xi64>)
                        outs(%init : tensor<3x5xi32>) : tensor<3x5xi32>
  return %r : tensor<3x5xi32>
}

// -----

//===----------------------------------------------------------------------===//
// A broadcast feeding an anchor, in the shape hip-fuse-rocmlir produces.
//===----------------------------------------------------------------------===//

// The point of converting the expand at all: a bias broadcast up to the matmul
// operand shape has to become TOSA alongside the anchor, or the broadcast stays
// a separate kernel and the fusion is lost. The ub.poison context and the
// tensor.empty outs buffers this pass does not claim survive untouched.
// CHECK-LABEL: func.func @rocMlir0
// CHECK: ub.poison : !hip.context
// CHECK: tensor.empty() : tensor<8x64x64xf32>
// CHECK: %[[ONES:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<8x64x64xf32>}>
// CHECK: %[[BCAST:.*]] = tosa.mul %arg0, %[[ONES]], %{{.*}} -> tensor<8x64x64xf32>
// CHECK: %[[A:.*]] = tosa.reshape %[[BCAST]], %{{.*}} -> tensor<1x512x64xf32>
// CHECK: tosa.matmul %[[A]], %{{.*}}
// CHECK-NOT: hip.expand
// CHECK-NOT: hip.matmul
func.func @rocMlir0(%bias: tensor<1x64x1xf32>, %s: tensor<3xi64>,
                    %w: tensor<64x64xf32>) -> tensor<8x64x64xf32>
    attributes {rock.arch = "gfx1151", rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e_init = tensor.empty() : tensor<8x64x64xf32>
  %e = hip.expand(%ctx) ins(%bias, %s : tensor<1x64x1xf32>, tensor<3xi64>)
                        outs(%e_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  %m_init = tensor.empty() : tensor<8x64x64xf32>
  %m = hip.matmul(%ctx) ins(%e, %w : tensor<8x64x64xf32>, tensor<64x64xf32>)
                        outs(%m_init : tensor<8x64x64xf32>) : tensor<8x64x64xf32>
  return %m : tensor<8x64x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Forms left alone.
//===----------------------------------------------------------------------===//

// The ones constant has to be built at the result shape, so a dynamic result has
// no TOSA spelling. Unlike the other hip ops this pass claims, that is not an
// error: OnnxToHip supports a dynamically shaped Expand on purpose, reading the
// extents back from the device, and this form has to stay a wrap_expand.
// CHECK-LABEL: func.func @expand_dynamic_result
// CHECK: hip.expand
// CHECK-NOT: tosa.mul
func.func @expand_dynamic_result(%ctx: !hip.context, %x: tensor<1x?xf32>,
                                 %s: tensor<2xi64>, %init: tensor<4x?xf32>)
    -> tensor<4x?xf32> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<1x?xf32>, tensor<2xi64>)
                        outs(%init : tensor<4x?xf32>) : tensor<4x?xf32>
  return %r : tensor<4x?xf32>
}

// -----

// A dynamic input is equally unconvertible: whether a dimension broadcasts or
// passes through cannot be decided here.
// CHECK-LABEL: func.func @expand_dynamic_input
// CHECK: hip.expand
// CHECK-NOT: tosa.mul
func.func @expand_dynamic_input(%ctx: !hip.context, %x: tensor<?x4xf32>,
                                %s: tensor<2xi64>, %init: tensor<3x4xf32>)
    -> tensor<3x4xf32> attributes {rock.kernel} {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<?x4xf32>, tensor<2xi64>)
                        outs(%init : tensor<3x4xf32>) : tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}

// -----

// Only outlined kernels are rewritten; the host graph keeps its runtime ops.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: hip.expand
// CHECK-NOT: tosa.mul
func.func @not_a_kernel(%ctx: !hip.context, %x: tensor<1x4xf32>,
                        %s: tensor<2xi64>, %init: tensor<3x4xf32>)
    -> tensor<3x4xf32> {
  %r = hip.expand(%ctx) ins(%x, %s : tensor<1x4xf32>, tensor<2xi64>)
                        outs(%init : tensor<3x4xf32>) : tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}
