// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file --verify-diagnostics %s | FileCheck %s

// ============================================================================
// TEST PURPOSE:
// Verify hip.resize lowers to tosa.resize and hip.constant to tosa.const
// inside a rock.kernel function.
//
// The substance of the resize conversion is two mismatches. hip.resize is
// (N, C, D_1..D_k) with the spatial axes trailing, while tosa.resize is 4-D
// NHWC with exactly two of them, so only k == 2 maps and it needs a transpose
// on each side. And hip.resize names a coordinate_transformation_mode, while
// TOSA takes an integer rational scale with an offset and a border:
//
//   in = (o * scale_d + offset) / scale_n
//   out - 1 == ((in_extent - 1) * scale_n - offset + border) / scale_d
//
// Each ONNX mode is an affine map from the output index to the input
// coordinate, so each is a choice of scale and offset; the border is then
// whatever makes TOSA's derived extent come back as the one hip.resize asked
// for. Writing OUT and IN for one axis's extents:
//
//   asymmetric      n = OUT,   d = IN,   offset = 0
//   half_pixel      n = 2*OUT, d = 2*IN, offset = IN - OUT
//   align_corners   n = OUT-1, d = IN-1, offset = 0
//
// half_pixel doubles the ratio because its offset is a half-integer otherwise,
// and TOSA takes integers only. The scale, offset and border values are
// checked exactly below, since they are where the coordinate math either holds
// or quietly shifts the image by half a pixel.
//
// hip.constant is a carrier for hip-externalize-constants and only its inline
// form has data to hand to TOSA. It is coverage rather than a live path: the
// ONNX-to-HIP pipeline externalizes every carrier and VerifyNoConstantCarriers
// then fails the compile if one survived, so a carrier does not normally reach
// this pass.
//
// FILE LAYOUT:
// Converting cases first, then the forms the pass declines. Declining is not
// an error -- both ops are dynamically legal, so an unsupported form stays a
// hip op instead of failing the pass.
// ============================================================================

//===----------------------------------------------------------------------===//
// hip.resize.
//===----------------------------------------------------------------------===//

// 2x bilinear upsample, half_pixel. IN 16 -> OUT 32 gives n = 64, d = 32,
// offset = 16 - 32 = -16, and a border of (32-1)*32 + -16 - (16-1)*64 = 16.
// The two transposes are the NCHW/NHWC bridge.
// CHECK-LABEL: func.func @resize_half_pixel
// CHECK: %[[NHWC:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x3x16x16xf16>) -> tensor<1x16x16x3xf16>
// CHECK: %[[SCALE:.*]] = tosa.const_shape {values = dense<[64, 32, 64, 32]> : tensor<4xindex>}
// CHECK: %[[OFF:.*]] = tosa.const_shape {values = dense<-16> : tensor<2xindex>}
// CHECK: %[[BORDER:.*]] = tosa.const_shape {values = dense<16> : tensor<2xindex>}
// CHECK: %[[R:.*]] = tosa.resize %[[NHWC]], %[[SCALE]], %[[OFF]], %[[BORDER]] {mode = BILINEAR} : {{.*}} -> tensor<1x32x32x3xf16>
// CHECK: tosa.transpose %[[R]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x3xf16>) -> tensor<1x3x32x32xf16>
// CHECK-NOT: hip.resize
func.func @resize_half_pixel(%ctx: !hip.context, %x: tensor<1x3x16x16xf16>,
                             %init: tensor<1x3x32x32xf16>)
    -> tensor<1x3x32x32xf16> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf16>)
                        outs(%init : tensor<1x3x32x32xf16>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xf16>
  return %r : tensor<1x3x32x32xf16>
}

// -----

// asymmetric needs no doubling, so the ratio stays 32/16 and the offset is
// zero. The border is (32-1)*16 - (16-1)*32 = 16.
// CHECK-LABEL: func.func @resize_asymmetric
// CHECK: %[[SCALE:.*]] = tosa.const_shape {values = dense<[32, 16, 32, 16]> : tensor<4xindex>}
// CHECK: %[[OFF:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>}
// CHECK: %[[BORDER:.*]] = tosa.const_shape {values = dense<16> : tensor<2xindex>}
// CHECK: tosa.resize %{{.*}}, %[[SCALE]], %[[OFF]], %[[BORDER]] {mode = BILINEAR}
// CHECK-NOT: hip.resize
func.func @resize_asymmetric(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                             %init: tensor<1x3x32x32xf32>)
    -> tensor<1x3x32x32xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x3x32x32xf32>)
       {mode = 1 : i64, coord_transform = 1 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xf32>
  return %r : tensor<1x3x32x32xf32>
}

// -----

// align_corners pins both endpoints, so the ratio is (OUT-1)/(IN-1) = 31/15
// and the border falls out as exactly zero.
// The offset and border are both zero here, so this case also pins the order
// the three shape operands are emitted in -- an ambiguous binding would let
// them swap unnoticed.
// CHECK-LABEL: func.func @resize_align_corners
// CHECK: %[[SCALE:.*]] = tosa.const_shape {values = dense<[31, 15, 31, 15]> : tensor<4xindex>}
// CHECK: %[[OFF:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>}
// CHECK: %[[BORDER:.*]] = tosa.const_shape {values = dense<0> : tensor<2xindex>}
// CHECK: tosa.resize %{{.*}}, %[[SCALE]], %[[OFF]], %[[BORDER]] {mode = BILINEAR}
// CHECK-NOT: hip.resize
func.func @resize_align_corners(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                                %init: tensor<1x3x32x32xf32>)
    -> tensor<1x3x32x32xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x3x32x32xf32>)
       {mode = 1 : i64, coord_transform = 2 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xf32>
  return %r : tensor<1x3x32x32xf32>
}

// -----

// Downsampling runs the same arithmetic with the extents swapped, which turns
// the border negative: (2-1)*8 + 2 - (4-1)*4 = -2. Nothing rejects that, and
// it is what makes TOSA derive an output extent of 2.
// CHECK-LABEL: func.func @resize_downsample
// CHECK: %[[SCALE:.*]] = tosa.const_shape {values = dense<[4, 8, 4, 8]> : tensor<4xindex>}
// CHECK: %[[OFF:.*]] = tosa.const_shape {values = dense<2> : tensor<2xindex>}
// CHECK: %[[BORDER:.*]] = tosa.const_shape {values = dense<-2> : tensor<2xindex>}
// CHECK: tosa.resize %{{.*}}, %[[SCALE]], %[[OFF]], %[[BORDER]] {mode = BILINEAR}
// CHECK-NOT: hip.resize
func.func @resize_downsample(%ctx: !hip.context, %x: tensor<2x8x4x4xf32>,
                             %init: tensor<2x8x2x2xf32>)
    -> tensor<2x8x2x2xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<2x8x4x4xf32>)
                        outs(%init : tensor<2x8x2x2xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<2x8x2x2xf32>
  return %r : tensor<2x8x2x2xf32>
}

// -----

// The two spatial axes are planned independently, so a resize that scales
// height and leaves width alone gets a mixed scale. Width 4 -> 4 under
// half_pixel is n = d = 8 with a zero offset and border.
// CHECK-LABEL: func.func @resize_height_only
// CHECK: %[[SCALE:.*]] = tosa.const_shape {values = dense<[16, 8, 8, 8]> : tensor<4xindex>}
// CHECK: %[[OFF:.*]] = tosa.const_shape {values = dense<[-4, 0]> : tensor<2xindex>}
// CHECK: %[[BORDER:.*]] = tosa.const_shape {values = dense<[4, 0]> : tensor<2xindex>}
// CHECK: tosa.resize %{{.*}}, %[[SCALE]], %[[OFF]], %[[BORDER]] {mode = BILINEAR}
// CHECK-NOT: hip.resize
func.func @resize_height_only(%ctx: !hip.context, %x: tensor<1x2x4x4xf32>,
                              %init: tensor<1x2x8x4xf32>)
    -> tensor<1x2x8x4xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x2x4x4xf32>)
                        outs(%init : tensor<1x2x8x4xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x4xf32>
  return %r : tensor<1x2x8x4xf32>
}

// -----

// nearest_mode is ignored for bilinear resize, so a nonzero value still lowers.
// CHECK-LABEL: func.func @resize_bilinear_ignores_nearest_mode
// CHECK: tosa.resize %{{.*}} {mode = BILINEAR}
// CHECK-NOT: hip.resize
func.func @resize_bilinear_ignores_nearest_mode(
    %ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
    %init: tensor<1x3x32x32xf32>)
    -> tensor<1x3x32x32xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x3x32x32xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 7 : i64}
       : tensor<1x3x32x32xf32>
  return %r : tensor<1x3x32x32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// hip.resize forms the pass declines.
//===----------------------------------------------------------------------===//

// TOSA's NEAREST_NEIGHBOR breaks a tie upward and ONNX round_prefer_floor
// breaks it downward, and the tie is reachable -- an asymmetric 2x upsample
// lands exactly halfway on every odd output index -- so nearest would disagree
// with the runtime on half the output rather than in the last bit.
// CHECK-LABEL: func.func @resize_reject_nearest
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @resize_reject_nearest(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                                 %init: tensor<1x3x32x32xf32>)
    -> tensor<1x3x32x32xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x3x32x32xf32>)
       {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xf32>
  return %r : tensor<1x3x32x32xf32>
}

// -----

// TOSA's integer BILINEAR leaves the result scaled by scale_y_n * scale_x_n
// for a following rescale to undo, and this pass emits no such rescale.
// CHECK-LABEL: func.func @resize_reject_integer
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @resize_reject_integer(%ctx: !hip.context, %x: tensor<1x3x16x16xi8>,
                                 %init: tensor<1x3x32x32xi8>)
    -> tensor<1x3x32x32xi8> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xi8>)
                        outs(%init : tensor<1x3x32x32xi8>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xi8>
  return %r : tensor<1x3x32x32xi8>
}

// -----

// tosa.resize resizes H and W of a 4-D tensor, so a volumetric resize with
// three spatial axes has no spelling.
// CHECK-LABEL: func.func @resize_reject_3d_spatial
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @resize_reject_3d_spatial(%ctx: !hip.context,
                                    %x: tensor<1x2x4x4x4xf32>,
                                    %init: tensor<1x2x8x8x8xf32>)
    -> tensor<1x2x8x8x8xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x2x4x4x4xf32>)
                        outs(%init : tensor<1x2x8x8x8xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x8x8xf32>
  return %r : tensor<1x2x8x8x8xf32>
}

// -----

// align_corners divides by IN-1 and OUT-1, so a unit extent puts a zero in the
// ratio and tosa.resize requires every scale value to be positive.
// CHECK-LABEL: func.func @resize_reject_align_corners_unit
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @resize_reject_align_corners_unit(%ctx: !hip.context,
                                            %x: tensor<1x2x1x4xf32>,
                                            %init: tensor<1x2x8x8xf32>)
    -> tensor<1x2x8x8xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x2x1x4xf32>)
                        outs(%init : tensor<1x2x8x8xf32>)
       {mode = 1 : i64, coord_transform = 2 : i64, nearest_mode = 0 : i64}
       : tensor<1x2x8x8xf32>
  return %r : tensor<1x2x8x8xf32>
}

// -----

// Only the spatial axes may move; a resize that changed the channel count
// would not be a resize.
// CHECK-LABEL: func.func @resize_reject_channel_change
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @resize_reject_channel_change(%ctx: !hip.context,
                                        %x: tensor<1x3x16x16xf32>,
                                        %init: tensor<1x4x32x32xf32>)
    -> tensor<1x4x32x32xf32> attributes {rock.kernel} {
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x4x32x32xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x4x32x32xf32>
  return %r : tensor<1x4x32x32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// hip.constant.
//===----------------------------------------------------------------------===//

// The inline form carries the data, so it becomes tosa.const with the same
// attribute.
// CHECK-LABEL: func.func @constant_inline_f32
// CHECK: "tosa.const"() <{values = dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>}>
// CHECK-NOT: hip.constant
func.func @constant_inline_f32() -> tensor<3xf32> attributes {rock.kernel} {
  %c = hip.constant {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>}
       : tensor<3xf32>
  return %c : tensor<3xf32>
}

// -----

// Integers take the same path; tosa.const is not float-only.
// CHECK-LABEL: func.func @constant_inline_i32
// CHECK: "tosa.const"() <{values = dense<7> : tensor<2x2xi32>}>
// CHECK-NOT: hip.constant
func.func @constant_inline_i32() -> tensor<2x2xi32> attributes {rock.kernel} {
  %c = hip.constant {value = dense<7> : tensor<2x2xi32>} : tensor<2x2xi32>
  return %c : tensor<2x2xi32>
}

// -----

// A file-backed carrier names a byte range that nothing has read yet, so there
// is no attribute to hand tosa.const. It stays a carrier for
// hip-externalize-constants, which is the pass that owns that decision.
// CHECK-LABEL: func.func @constant_reject_file_backed
// CHECK: hip.constant
// CHECK-NOT: tosa.const
func.func @constant_reject_file_backed() -> tensor<4xf32>
    attributes {rock.kernel} {
  %c = hip.constant {location = "weights.bin", offset = 0 : i64,
                     size = 16 : i64} : tensor<4xf32>
  return %c : tensor<4xf32>
}

// -----

// A memory-address carrier is process-local and equally unreadable here.
// CHECK-LABEL: func.func @constant_reject_memory_backed
// CHECK: hip.constant
// CHECK-NOT: tosa.const
func.func @constant_reject_memory_backed() -> tensor<4xf32>
    attributes {rock.kernel} {
  %c = hip.constant {memory_address = 140737488355328 : i64, size = 16 : i64}
       : tensor<4xf32>
  return %c : tensor<4xf32>
}

// -----

// The pass runs only inside a rock.kernel; everywhere else both ops are left
// for the paths that own them.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: hip.constant
// CHECK: hip.resize
// CHECK-NOT: tosa.resize
func.func @not_a_kernel(%ctx: !hip.context, %x: tensor<1x3x16x16xf32>,
                        %init: tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> {
  %c = hip.constant {value = dense<1.0> : tensor<3xf32>} : tensor<3xf32>
  %r = hip.resize(%ctx) ins(%x : tensor<1x3x16x16xf32>)
                        outs(%init : tensor<1x3x32x32xf32>)
       {mode = 1 : i64, coord_transform = 0 : i64, nearest_mode = 0 : i64}
       : tensor<1x3x32x32xf32>
  return %r : tensor<1x3x32x32xf32>
}
