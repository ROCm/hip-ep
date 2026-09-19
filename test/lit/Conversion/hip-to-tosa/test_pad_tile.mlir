// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s | FileCheck %s

// ============================================================================
// TEST PURPOSE:
// Verify hip.pad and hip.tile lower to tosa.pad and tosa.tile inside a
// rock.kernel function, so rocMLIR can absorb them into a fused kernel.
//
// Both ops carry their shape information as an operand rather than an
// attribute, because ONNX Pad and ONNX Tile take `pads` and `repeats` as
// inputs. TOSA wants a !tosa.shape instead, so each converts only when that
// operand is constant, and is left as a hip op otherwise.
//
// The pad layouts differ and the reordering is the substance of the
// conversion: ONNX groups all the begins then all the ends,
// [x0_begin, x1_begin, .., x0_end, x1_end, ..], while TOSA interleaves them
// per dimension, [d0_lo, d0_hi, d1_lo, d1_hi, ..].
//
// FILE LAYOUT:
// Converting cases first, then the forms the pass declines, then the hard
// rejections.
// ============================================================================

// CHECK-LABEL: func.func @tile_basic
// CHECK: %[[M:.*]] = tosa.const_shape {values = dense<[2, 3]> : tensor<2xindex>}
// CHECK: tosa.tile %arg1, %[[M]]
// CHECK-NOT: hip.tile
func.func @tile_basic(%ctx: !hip.context, %x: tensor<2x3xf32>,
                      %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<2x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<4x9xf32>) : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// A repeat of 1 leaves the dimension alone, which is still a valid tile.
// CHECK-LABEL: func.func @tile_identity_axis
// CHECK: %[[M:.*]] = tosa.const_shape {values = dense<[1, 4]> : tensor<2xindex>}
// CHECK: tosa.tile %arg1, %[[M]]
func.func @tile_identity_axis(%ctx: !hip.context, %x: tensor<2x3xf32>,
                              %init: tensor<2x12xf32>) -> tensor<2x12xf32>
    attributes {rock.kernel} {
  %repeats = arith.constant dense<[1, 4]> : tensor<2xi64>
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<2x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<2x12xf32>) : tensor<2x12xf32>
  return %r : tensor<2x12xf32>
}

// The ONNX pads [1, 3, 2, 4] mean dim0 gains 1 before and 2 after, dim1 gains
// 3 before and 4 after, so 1x2 becomes 4x9. TOSA spells the same thing as the
// interleaved [1, 2, 3, 4].
// CHECK-LABEL: func.func @pad_constant
// CHECK: %[[PC:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}>
// CHECK: %[[P:.*]] = tosa.const_shape {values = dense<[1, 2, 3, 4]> : tensor<4xindex>}
// CHECK: tosa.pad %arg1, %[[P]], %[[PC]]
// CHECK-NOT: hip.pad
func.func @pad_constant(%ctx: !hip.context, %x: tensor<1x2xf32>,
                        %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "constant"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// mode defaults to "constant", so an op without the attribute still converts.
// CHECK-LABEL: func.func @pad_default_mode
// CHECK: tosa.pad
func.func @pad_default_mode(%ctx: !hip.context, %x: tensor<1x2xf32>,
                            %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     outs(%init : tensor<4x9xf32>) : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// `axes` restricts the pads to the listed dimensions; the rest stay at zero,
// so a single-axis pad still produces a full-rank TOSA padding.
// CHECK-LABEL: func.func @pad_axes_subset
// CHECK: %[[P:.*]] = tosa.const_shape {values = dense<[0, 0, 2, 3]> : tensor<4xindex>}
// CHECK: tosa.pad %arg1, %[[P]]
func.func @pad_axes_subset(%ctx: !hip.context, %x: tensor<2x3xf32>,
                           %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[2, 3]> : tensor<2xi64>
  %axes = arith.constant dense<[1]> : tensor<1xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<2x3xf32>, tensor<2xi64>)
                     axes(%axes : tensor<1xi64>)
                     outs(%init : tensor<2x8xf32>)
                     {mode = "constant"} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// A negative axis counts from the end, matching ONNX.
// CHECK-LABEL: func.func @pad_negative_axis
// CHECK: %[[P:.*]] = tosa.const_shape {values = dense<[0, 0, 2, 3]> : tensor<4xindex>}
// CHECK: tosa.pad %arg1, %[[P]]
func.func @pad_negative_axis(%ctx: !hip.context, %x: tensor<2x3xf32>,
                             %init: tensor<2x8xf32>) -> tensor<2x8xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[2, 3]> : tensor<2xi64>
  %axes = arith.constant dense<[-1]> : tensor<1xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<2x3xf32>, tensor<2xi64>)
                     axes(%axes : tensor<1xi64>)
                     outs(%init : tensor<2x8xf32>)
                     {mode = "constant"} : tensor<2x8xf32>
  return %r : tensor<2x8xf32>
}

// An empty `axes` means every dimension, not none. wrap_pad picks the per-axis
// pads layout on axes_host.empty(), and it populates axes_host only when the
// operand is present and non-empty, so a zero-length `axes` reaches the same
// branch an omitted one does and `pads` carries the full 2 * rank entries.
// CHECK-LABEL: func.func @pad_empty_axes
// CHECK: %[[P:.*]] = tosa.const_shape {values = dense<[1, 1, 2, 3]> : tensor<4xindex>}
// CHECK: tosa.pad %arg1, %[[P]]
// CHECK-NOT: hip.pad
func.func @pad_empty_axes(%ctx: !hip.context, %x: tensor<2x3xf32>,
                          %init: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 2, 1, 3]> : tensor<4xi64>
  %axes = arith.constant dense<> : tensor<0xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<2x3xf32>, tensor<4xi64>)
                     axes(%axes : tensor<0xi64>)
                     outs(%init : tensor<4x8xf32>)
                     {mode = "constant"} : tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// The fill value becomes tosa.pad's one-element pad_const operand.
// CHECK-LABEL: func.func @pad_with_cval
// CHECK: %[[PC:.*]] = "tosa.const"() <{values = dense<2.500000e+00> : tensor<1xf32>}>
// CHECK: tosa.pad %arg1, %{{.*}}, %[[PC]]
func.func @pad_with_cval(%ctx: !hip.context, %x: tensor<1x2xf32>,
                         %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %cval = arith.constant dense<2.500000e+00> : tensor<f32>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     cval(%cval : tensor<f32>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "constant"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// Integer tensors pad the same way, with an integer pad_const.
// CHECK-LABEL: func.func @pad_integer
// CHECK: %[[PC:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}>
// CHECK: tosa.pad %arg1, %{{.*}}, %[[PC]]
func.func @pad_integer(%ctx: !hip.context, %x: tensor<4xi32>,
                       %init: tensor<7xi32>) -> tensor<7xi32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 2]> : tensor<2xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<4xi32>, tensor<2xi64>)
                     outs(%init : tensor<7xi32>)
                     {mode = "constant"} : tensor<7xi32>
  return %r : tensor<7xi32>
}

//===----------------------------------------------------------------------===//
// The rock.kernel guard, which is a property of the pass rather than of any
// one op.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @tile_not_a_kernel
// CHECK: hip.tile
// CHECK-NOT: tosa.tile
func.func @tile_not_a_kernel(%ctx: !hip.context, %x: tensor<2x3xf32>,
                             %init: tensor<4x9xf32>) -> tensor<4x9xf32> {
  %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<2x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<4x9xf32>) : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Forms this pass does not claim. Both ops are legal by operand and mode
// rather than outright, so an unsupported one stays a hip op for the runtime
// lowering and the pass still succeeds. These share one chunk since none
// produce a diagnostic.
//===----------------------------------------------------------------------===//

// TOSA has no reflect, edge or wrap pad, only the constant fill.
// CHECK-LABEL: func.func @pad_reflect
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_reflect(%ctx: !hip.context, %x: tensor<1x2xf32>,
                       %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "reflect"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// CHECK-LABEL: func.func @pad_edge
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_edge(%ctx: !hip.context, %x: tensor<1x2xf32>,
                    %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "edge"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// !tosa.shape is built from the operand's values, so a computed `pads` has no
// spelling here.
// CHECK-LABEL: func.func @pad_dynamic_pads
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_dynamic_pads(%ctx: !hip.context, %x: tensor<1x2xf32>,
                            %pads: tensor<4xi64>, %init: tensor<4x9xf32>)
    -> tensor<4x9xf32> attributes {rock.kernel} {
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "constant"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// ONNX opset 18 allows negative pads as a crop. The result dimension then
// stops following from input + lo + hi, which is the invariant the shape check
// relies on, so those are declined rather than half-supported.
// CHECK-LABEL: func.func @pad_negative_pads
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_negative_pads(%ctx: !hip.context, %x: tensor<1x4xf32>,
                             %init: tensor<1x3xf32>) -> tensor<1x3xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[0, -1, 0, 0]> : tensor<4xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x4xf32>, tensor<4xi64>)
                     outs(%init : tensor<1x3xf32>)
                     {mode = "constant"} : tensor<1x3xf32>
  return %r : tensor<1x3xf32>
}

// ONNX requires the entries of `axes` to be distinct. A repeat would drop one
// of the two paddings, and it has to be caught by axis rather than by reading
// the accumulated pads back, since the earlier entry may itself be (0, 0).
// CHECK-LABEL: func.func @pad_repeated_axis
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_repeated_axis(%ctx: !hip.context, %x: tensor<2x3xf32>,
                             %init: tensor<7x3xf32>) -> tensor<7x3xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[0, 2, 0, 3]> : tensor<4xi64>
  %axes = arith.constant dense<[0, 0]> : tensor<2xi64>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<2x3xf32>, tensor<4xi64>)
                     axes(%axes : tensor<2xi64>)
                     outs(%init : tensor<7x3xf32>)
                     {mode = "constant"} : tensor<7x3xf32>
  return %r : tensor<7x3xf32>
}

// The fill becomes a one-element tosa.pad operand, so a constant holding more
// than one distinct value has nothing to collapse to. The legality predicate
// has to decline it for the same reason the rewrite would, otherwise the op is
// marked illegal and the pass fails on it.
// CHECK-LABEL: func.func @pad_non_splat_cval
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_non_splat_cval(%ctx: !hip.context, %x: tensor<1x2xf32>,
                              %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 3, 2, 4]> : tensor<4xi64>
  %cval = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<1x2xf32>, tensor<4xi64>)
                     cval(%cval : tensor<2xf32>)
                     outs(%init : tensor<4x9xf32>)
                     {mode = "constant"} : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// A fill whose type does not match the tensor it fills is not read either.
// CHECK-LABEL: func.func @pad_mistyped_cval
// CHECK: hip.pad
// CHECK-NOT: tosa.pad
func.func @pad_mistyped_cval(%ctx: !hip.context, %x: tensor<4xi32>,
                             %init: tensor<7xi32>) -> tensor<7xi32>
    attributes {rock.kernel} {
  %pads = arith.constant dense<[1, 2]> : tensor<2xi64>
  %cval = arith.constant dense<2.500000e+00> : tensor<f32>
  %r = hip.pad(%ctx) ins(%x, %pads : tensor<4xi32>, tensor<2xi64>)
                     cval(%cval : tensor<f32>)
                     outs(%init : tensor<7xi32>)
                     {mode = "constant"} : tensor<7xi32>
  return %r : tensor<7xi32>
}

// A computed `repeats` cannot become a !tosa.shape either.
// CHECK-LABEL: func.func @tile_dynamic_repeats
// CHECK: hip.tile
// CHECK-NOT: tosa.tile
func.func @tile_dynamic_repeats(%ctx: !hip.context, %x: tensor<2x3xf32>,
                                %repeats: tensor<2xi64>,
                                %init: tensor<4x9xf32>) -> tensor<4x9xf32>
    attributes {rock.kernel} {
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<2x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<4x9xf32>) : tensor<4x9xf32>
  return %r : tensor<4x9xf32>
}

// A zero repeat empties the dimension, which TOSA's positive multiples cannot
// express.
// CHECK-LABEL: func.func @tile_zero_repeat
// CHECK: hip.tile
// CHECK-NOT: tosa.tile
func.func @tile_zero_repeat(%ctx: !hip.context, %x: tensor<2x3xf32>,
                            %init: tensor<0x3xf32>) -> tensor<0x3xf32>
    attributes {rock.kernel} {
  %repeats = arith.constant dense<[0, 1]> : tensor<2xi64>
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<2x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<0x3xf32>) : tensor<0x3xf32>
  return %r : tensor<0x3xf32>
}

// Dynamic shapes give neither pattern a shape to build from.
// CHECK-LABEL: func.func @tile_dynamic_shape
// CHECK: hip.tile
// CHECK-NOT: tosa.tile
func.func @tile_dynamic_shape(%ctx: !hip.context, %x: tensor<?x3xf32>,
                              %init: tensor<?x9xf32>) -> tensor<?x9xf32>
    attributes {rock.kernel} {
  %repeats = arith.constant dense<[2, 3]> : tensor<2xi64>
  %r = hip.tile(%ctx) ins(%x, %repeats : tensor<?x3xf32>, tensor<2xi64>)
                      outs(%init : tensor<?x9xf32>) : tensor<?x9xf32>
  return %r : tensor<?x9xf32>
}
