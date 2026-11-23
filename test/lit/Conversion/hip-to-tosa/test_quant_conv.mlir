// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify hip.quant_conv lowers to an i8 x i8 -> i32 tosa.conv2d with zero
// points 0, in the channel-last layouts, and that a ui8 input is rebiased to
// x - 128 through unsigned_cast so the kernel boundary becomes signless.
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @quant_conv_i8
// CHECK-SAME: (%[[X:.*]]: tensor<1x2x4x4xi8>, %[[W:.*]]: tensor<2x2x3x3xi8>) -> tensor<1x2x4x4xi32>
// CHECK-NOT: unsigned_cast
// CHECK: %[[XN:.*]] = tosa.transpose %[[X]] {perms = array<i32: 0, 2, 3, 1>}
// CHECK: %[[WN:.*]] = tosa.transpose %[[W]] {perms = array<i32: 0, 2, 3, 1>}
// CHECK: %[[C:.*]] = tosa.conv2d %[[XN]], %[[WN]]
// CHECK-SAME: acc_type = i32
// CHECK-SAME: group = 1 : i64
// CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
// CHECK-SAME: -> tensor<1x4x4x2xi32>
// CHECK: tosa.transpose %[[C]] {perms = array<i32: 0, 3, 1, 2>}
// CHECK-NOT: hip.quant_conv
func.func @quant_conv_i8(%x: tensor<1x2x4x4xi8>, %w: tensor<2x2x3x3xi8>)
    -> tensor<1x2x4x4xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x2x4x4xi32>
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xi8>,
         tensor<2x2x3x3xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

// CHECK-LABEL: func.func @quant_conv_ui8
// CHECK-SAME: (%[[X:.*]]: tensor<1x2x4x4xi8>,
// CHECK: %[[WIDE:.*]] = tosa.custom %[[X]] {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<1x2x4x4xi32>
// CHECK: %[[OFF:.*]] = "tosa.const"() <{values = dense<128> : tensor<1x1x1x1xi32>}>
// CHECK: %[[SUB:.*]] = tosa.sub %[[WIDE]], %[[OFF]]
// CHECK: %[[X8:.*]] = tosa.cast %[[SUB]] : (tensor<1x2x4x4xi32>) -> tensor<1x2x4x4xi8>
// CHECK: %[[XN:.*]] = tosa.transpose %[[X8]]
// CHECK: tosa.conv2d %[[XN]]
// CHECK-SAME: acc_type = i32
// CHECK-NOT: ui8
func.func @quant_conv_ui8(%x: tensor<1x2x4x4xui8>, %w: tensor<2x2x3x3xi8>)
    -> tensor<1x2x4x4xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x2x4x4xi32>
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x2x4x4xui8>,
         tensor<2x2x3x3xi8>) outs(%e : tensor<1x2x4x4xi32>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x2x4x4xi32>
  return %r : tensor<1x2x4x4xi32>
}

// -----

// A depthwise stride-2 conv keeps its group, and the trailing pad that the
// floored ONNX window does not read is dropped.
// CHECK-LABEL: func.func @quant_conv_depthwise
// CHECK: tosa.conv2d
// CHECK-SAME: acc_type = i32
// CHECK-SAME: group = 4 : i64
// CHECK-SAME: pad = array<i64: 1, 0, 1, 0>
// CHECK-SAME: stride = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x3x3x4xi32>
func.func @quant_conv_depthwise(%x: tensor<1x4x6x6xui8>,
                                %w: tensor<4x1x3x3xi8>)
    -> tensor<1x4x3x3xi32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e = tensor.empty() : tensor<1x4x3x3xi32>
  %r = hip.quant_conv(%ctx) ins(%x, %w : tensor<1x4x6x6xui8>,
         tensor<4x1x3x3xi8>) outs(%e : tensor<1x4x3x3xi32>)
         {dilations = [1, 1], group = 4 : i64, kernel_shape = [3, 3],
          pads = [1, 1, 1, 1], strides = [2, 2]} : tensor<1x4x3x3xi32>
  return %r : tensor<1x4x3x3xi32>
}
