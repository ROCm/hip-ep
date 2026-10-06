// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.pool / hip.global_pool lower inside a rock.kernel function.
// TOSA pooling is 2D and undilated, so the conversion normalizes what it can
// onto tosa.max_pool2d / tosa.avg_pool2d and decomposes the rest; global
// pooling becomes keepdims reductions.
//
// AVERAGE divisors: tosa.avg_pool2d divides by the window's in-bounds
// coverage of its own pad attribute, which is ONNX count_include_pad=0. The
// other two divisors materialize the padding as zeros and pool unpadded so
// the divisor is the full kernel volume.
//
// FILE LAYOUT:
// Everything is in the first --split-input-file chunk. A configuration TOSA
// cannot express stays a hip.pool that ends the fusion chain, so there are no
// legalization failures to isolate.
//
// This test validates:
// - 2D max / average map to the native TOSA pools through NCHW<->NHWC
// - padded average picks the divisor ONNX asked for
// - a 1D window pools as a 2D one over a unit spatial dim
// - LP pooling decomposes to pow / avg_pool2d / pow
// - global average / max / LP become keepdims reductions
// - 3D windows, dilation and MaxPool Indices stay hip.pool
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @max_pool2d
// CHECK: %[[NHWC:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x3x32x32xf16>) -> tensor<1x32x32x3xf16>
// CHECK: %[[POOL:.*]] = tosa.max_pool2d %[[NHWC]] {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x32x32x3xf16>) -> tensor<1x16x16x3xf16>
// CHECK: tosa.transpose %[[POOL]] {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x16x16x3xf16>) -> tensor<1x3x16x16xf16>
// CHECK-NOT: hip.pool
func.func @max_pool2d(%ctx: !hip.context, %x: tensor<1x3x32x32xf16>,
                      %init: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16>
    attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x32x32xf16>)
                      outs(%init : tensor<1x3x16x16xf16>)
                      {ceil_mode = 0 : i64, dilations = [1, 1],
                       kernel_shape = [2, 2], pads = [0, 0, 0, 0],
                       pool_mode = 1 : i64, storage_order = 0 : i64,
                       strides = [2, 2]} : tensor<1x3x16x16xf16>
  return %y : tensor<1x3x16x16xf16>
}

// CHECK-LABEL: func.func @avg_pool2d
// CHECK: %[[NHWC:.*]] = tosa.transpose %arg1 {perms = array<i32: 0, 2, 3, 1>}
// CHECK: tosa.avg_pool2d %[[NHWC]]
// CHECK-SAME: acc_type = f32
// CHECK-SAME: kernel = array<i64: 3, 3>
// CHECK-SAME: pad = array<i64: 0, 0, 0, 0>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK: tosa.transpose
// CHECK-NOT: hip.pool
func.func @avg_pool2d(%ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
                      %init: tensor<1x3x6x6xf32>) -> tensor<1x3x6x6xf32>
    attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x8x8xf32>)
                      outs(%init : tensor<1x3x6x6xf32>)
                      {ceil_mode = 0 : i64, count_include_pad = 0 : i64,
                       dilations = [1, 1], kernel_shape = [3, 3],
                       pads = [0, 0, 0, 0], pool_mode = 0 : i64,
                       storage_order = 0 : i64, strides = [1, 1]}
                      : tensor<1x3x6x6xf32>
  return %y : tensor<1x3x6x6xf32>
}

// count_include_pad=0 is what tosa.avg_pool2d already does, so the padding
// stays on the pool.
// CHECK-LABEL: func.func @avg_pool2d_exclude_pad
// CHECK-NOT: tosa.pad
// CHECK: tosa.avg_pool2d
// CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
// CHECK-NOT: hip.pool
func.func @avg_pool2d_exclude_pad(%ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
                                  %init: tensor<1x3x8x8xf32>)
    -> tensor<1x3x8x8xf32> attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x8x8xf32>)
                      outs(%init : tensor<1x3x8x8xf32>)
                      {ceil_mode = 0 : i64, count_include_pad = 0 : i64,
                       dilations = [1, 1], kernel_shape = [3, 3],
                       pads = [1, 1, 1, 1], pool_mode = 0 : i64,
                       storage_order = 0 : i64, strides = [1, 1]}
                      : tensor<1x3x8x8xf32>
  return %y : tensor<1x3x8x8xf32>
}

// count_include_pad=1 divides by the full kernel volume, so the pad becomes
// explicit zeros and the pool itself is unpadded.
// CHECK-LABEL: func.func @avg_pool2d_include_pad
// CHECK: %[[PAD:.*]] = tosa.pad %{{.*}} -> tensor<1x10x10x3xf32>
// CHECK: tosa.avg_pool2d %[[PAD]]
// CHECK-SAME: pad = array<i64: 0, 0, 0, 0>
// CHECK-NOT: hip.pool
func.func @avg_pool2d_include_pad(%ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
                                  %init: tensor<1x3x8x8xf32>)
    -> tensor<1x3x8x8xf32> attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x8x8xf32>)
                      outs(%init : tensor<1x3x8x8xf32>)
                      {ceil_mode = 0 : i64, count_include_pad = 1 : i64,
                       dilations = [1, 1], kernel_shape = [3, 3],
                       pads = [1, 1, 1, 1], pool_mode = 0 : i64,
                       storage_order = 0 : i64, strides = [1, 1]}
                      : tensor<1x3x8x8xf32>
  return %y : tensor<1x3x8x8xf32>
}

// A 1D window gets a unit leading spatial dim so the 2D pool can take it.
// CHECK-LABEL: func.func @max_pool1d
// CHECK: tosa.reshape %arg1, %{{.*}} -> tensor<1x3x1x32xf32>
// CHECK: tosa.max_pool2d
// CHECK-SAME: kernel = array<i64: 1, 2>
// CHECK-SAME: stride = array<i64: 1, 1>
// CHECK: tosa.reshape %{{.*}} -> tensor<1x3x31xf32>
// CHECK-NOT: hip.pool
func.func @max_pool1d(%ctx: !hip.context, %x: tensor<1x3x32xf32>,
                      %init: tensor<1x3x31xf32>) -> tensor<1x3x31xf32>
    attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x32xf32>)
                      outs(%init : tensor<1x3x31xf32>)
                      {ceil_mode = 0 : i64, dilations = [1],
                       kernel_shape = [2], pads = [0, 0],
                       pool_mode = 1 : i64, storage_order = 0 : i64,
                       strides = [1]} : tensor<1x3x31xf32>
  return %y : tensor<1x3x31xf32>
}

// LP takes the window sum, which TOSA only has as an average, so the mean is
// scaled back up by the kernel volume.
// CHECK-LABEL: func.func @lp_pool2d
// CHECK: tosa.abs
// CHECK: tosa.pow
// CHECK: tosa.avg_pool2d
// CHECK: tosa.mul
// CHECK: tosa.pow
// CHECK-NOT: hip.pool
func.func @lp_pool2d(%ctx: !hip.context, %x: tensor<1x3x4x4xf32>,
                     %init: tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x4x4xf32>)
                      outs(%init : tensor<1x3x2x2xf32>)
                      {ceil_mode = 0 : i64, dilations = [1, 1],
                       kernel_shape = [2, 2], p = 2 : i64,
                       pads = [0, 0, 0, 0], pool_mode = 2 : i64,
                       storage_order = 0 : i64, strides = [2, 2]}
                      : tensor<1x3x2x2xf32>
  return %y : tensor<1x3x2x2xf32>
}

// CHECK-LABEL: func.func @global_avg_pool
// CHECK: tosa.reduce_sum
// CHECK-SAME: axis = 2
// CHECK: tosa.reduce_sum
// CHECK-SAME: axis = 3
// CHECK-NOT: hip.global_pool
func.func @global_avg_pool(%ctx: !hip.context, %x: tensor<1x3x5x5xf32>,
                           %init: tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
    attributes {rock.kernel} {
  %y = hip.global_pool(%ctx) ins(%x : tensor<1x3x5x5xf32>)
                             outs(%init : tensor<1x3x1x1xf32>)
                             {mode = 0 : i64, p = 2 : i64}
                             : tensor<1x3x1x1xf32>
  return %y : tensor<1x3x1x1xf32>
}

// CHECK-LABEL: func.func @global_max_pool
// CHECK: tosa.reduce_max
// CHECK-SAME: axis = 2
// CHECK: tosa.reduce_max
// CHECK-SAME: axis = 3
// CHECK-NOT: hip.global_pool
func.func @global_max_pool(%ctx: !hip.context, %x: tensor<1x3x5x5xf16>,
                           %init: tensor<1x3x1x1xf16>) -> tensor<1x3x1x1xf16>
    attributes {rock.kernel} {
  %y = hip.global_pool(%ctx) ins(%x : tensor<1x3x5x5xf16>)
                             outs(%init : tensor<1x3x1x1xf16>)
                             {mode = 1 : i64} : tensor<1x3x1x1xf16>
  return %y : tensor<1x3x1x1xf16>
}

// CHECK-LABEL: func.func @global_lp_pool
// CHECK: tosa.abs
// CHECK: tosa.pow
// CHECK: tosa.reduce_sum
// CHECK-SAME: axis = 2
// CHECK: tosa.reduce_sum
// CHECK-SAME: axis = 3
// CHECK: tosa.pow
// CHECK-NOT: hip.global_pool
func.func @global_lp_pool(%ctx: !hip.context, %x: tensor<1x3x4x4xf32>,
                          %init: tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
    attributes {rock.kernel} {
  %y = hip.global_pool(%ctx) ins(%x : tensor<1x3x4x4xf32>)
                             outs(%init : tensor<1x3x1x1xf32>)
                             {mode = 2 : i64, p = 2 : i64}
                             : tensor<1x3x1x1xf32>
  return %y : tensor<1x3x1x1xf32>
}

//===----------------------------------------------------------------------===//
// TOSA pooling is 2D, dense, and single-result, so these stay hip.pool.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @max_pool3d_stays_hip
// CHECK: hip.pool
// CHECK-NOT: tosa.max_pool2d
func.func @max_pool3d_stays_hip(%ctx: !hip.context, %x: tensor<1x2x4x4x4xf32>,
                                %init: tensor<1x2x2x2x2xf32>)
    -> tensor<1x2x2x2x2xf32> attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x2x4x4x4xf32>)
                      outs(%init : tensor<1x2x2x2x2xf32>)
                      {ceil_mode = 0 : i64, dilations = [1, 1, 1],
                       kernel_shape = [2, 2, 2], pads = [0, 0, 0, 0, 0, 0],
                       pool_mode = 1 : i64, storage_order = 0 : i64,
                       strides = [2, 2, 2]} : tensor<1x2x2x2x2xf32>
  return %y : tensor<1x2x2x2x2xf32>
}

// CHECK-LABEL: func.func @dilated_pool_stays_hip
// CHECK: hip.pool
// CHECK-NOT: tosa.max_pool2d
func.func @dilated_pool_stays_hip(%ctx: !hip.context, %x: tensor<1x3x8x8xf32>,
                                  %init: tensor<1x3x6x6xf32>)
    -> tensor<1x3x6x6xf32> attributes {rock.kernel} {
  %y = hip.pool(%ctx) ins(%x : tensor<1x3x8x8xf32>)
                      outs(%init : tensor<1x3x6x6xf32>)
                      {ceil_mode = 0 : i64, dilations = [2, 2],
                       kernel_shape = [2, 2], pads = [0, 0, 0, 0],
                       pool_mode = 1 : i64, storage_order = 0 : i64,
                       strides = [1, 1]} : tensor<1x3x6x6xf32>
  return %y : tensor<1x3x6x6xf32>
}

// CHECK-LABEL: func.func @pool_indices_stays_hip
// CHECK: hip.pool
// CHECK-NOT: tosa.max_pool2d
func.func @pool_indices_stays_hip(%ctx: !hip.context, %x: tensor<1x3x32x32xf32>,
                                  %y_init: tensor<1x3x16x16xf32>,
                                  %i_init: tensor<1x3x16x16xi64>)
    -> (tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>)
    attributes {rock.kernel} {
  %y, %idx = hip.pool(%ctx) ins(%x : tensor<1x3x32x32xf32>)
      outs(%y_init, %i_init : tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>)
      {ceil_mode = 0 : i64, dilations = [1, 1], kernel_shape = [2, 2],
       pads = [0, 0, 0, 0], pool_mode = 1 : i64, storage_order = 0 : i64,
       strides = [2, 2]} : tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>
  return %y, %idx : tensor<1x3x16x16xf32>, tensor<1x3x16x16xi64>
}
