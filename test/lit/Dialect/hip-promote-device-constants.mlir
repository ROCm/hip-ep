// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-promote-device-constants: a tensor arith.constant bufferizes to a host
// memref.global, so every device read of it -- a hip op operand or a
// tensor.insert_slice source/destination, directly or through a view -- moves
// to one hip.constant, which externalization places in device memory.
// Host uses keep the arith.constant.
//
// RUN: hip-mlir-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(func.func(hip-promote-device-constants))' \
// RUN:   %s | FileCheck %s

// Three hip ops read one integer fill; they share a single device copy.
// CHECK-LABEL: func.func @direct_reads_share_one_device_constant
// CHECK:         %[[D:.*]] = hip.constant {value = dense<1> : tensor<3xi64>} : tensor<3xi64>
// CHECK-NOT:     hip.constant
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %[[D]] :
// CHECK:         hip.gather(%{{.*}}) ins(%{{.*}}, %[[D]] :
// CHECK:         hip.cast(%{{.*}}) ins(%[[D]] :
func.func @direct_reads_share_one_device_constant(
    %ctx: !hip.context, %x: tensor<3xi64>, %t: tensor<8x4xf32>)
    -> (tensor<3xi64>, tensor<3x4xf32>, tensor<3xf32>) {
  %c = arith.constant dense<1> : tensor<3xi64>
  %e0 = tensor.empty() : tensor<3xi64>
  %a = hip.add(%ctx) ins(%x, %c : tensor<3xi64>, tensor<3xi64>)
      outs(%e0 : tensor<3xi64>) -> tensor<3xi64>
  %e1 = tensor.empty() : tensor<3x4xf32>
  %g = hip.gather(%ctx) ins(%t, %c : tensor<8x4xf32>, tensor<3xi64>)
      outs(%e1 : tensor<3x4xf32>) : tensor<3x4xf32>
  %e2 = tensor.empty() : tensor<3xf32>
  %f = hip.cast(%ctx) ins(%c : tensor<3xi64>) outs(%e2 : tensor<3xf32>)
      {to = 1 : i64} : tensor<3xf32>
  return %a, %g, %f : tensor<3xi64>, tensor<3x4xf32>, tensor<3xf32>
}

// -----

// Only the host reads it: nothing to promote.
// CHECK-LABEL: func.func @host_only_use_stays
// CHECK-NOT:     hip.constant
// CHECK:         %[[C:.*]] = arith.constant dense<[4, 5, 6]> : tensor<3xi64>
// CHECK:         tensor.extract %[[C]]
func.func @host_only_use_stays(%i: index) -> i64 {
  %c = arith.constant dense<[4, 5, 6]> : tensor<3xi64>
  %v = tensor.extract %c[%i] : tensor<3xi64>
  return %v : i64
}

// -----

// A host read and a hip read of the same constant: each keeps the form it
// needs.
// CHECK-LABEL: func.func @mixed_uses_split
// CHECK-DAG:     %[[C:.*]] = arith.constant dense<2.500000e-01> : tensor<4xf32>
// CHECK-DAG:     %[[D:.*]] = hip.constant {value = dense<2.500000e-01> : tensor<4xf32>} : tensor<4xf32>
// CHECK:         tensor.extract %[[C]]
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %[[D]] :
func.func @mixed_uses_split(%ctx: !hip.context, %x: tensor<4xf32>, %i: index)
    -> (f32, tensor<4xf32>) {
  %c = arith.constant dense<0.25> : tensor<4xf32>
  %s = tensor.extract %c[%i] : tensor<4xf32>
  %e = tensor.empty() : tensor<4xf32>
  %a = hip.add(%ctx) ins(%x, %c : tensor<4xf32>, tensor<4xf32>)
      outs(%e : tensor<4xf32>) -> tensor<4xf32>
  return %s, %a : f32, tensor<4xf32>
}

// -----

// A hip op reading through a reshape view reads the constant: the view is
// cloned onto the device copy.
// CHECK-LABEL: func.func @through_expand_shape
// CHECK:         %[[D:.*]] = hip.constant {value = dense<0> : tensor<3xi64>} : tensor<3xi64>
// CHECK:         %[[DV:.*]] = tensor.expand_shape %[[D]] {{\[\[0, 1\]\]}} output_shape [1, 3]
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %[[DV]] :
func.func @through_expand_shape(%ctx: !hip.context, %x: tensor<1x3xi64>)
    -> tensor<1x3xi64> {
  %c = arith.constant dense<0> : tensor<3xi64>
  %v = tensor.expand_shape %c [[0, 1]] output_shape [1, 3]
      : tensor<3xi64> into tensor<1x3xi64>
  %e = tensor.empty() : tensor<1x3xi64>
  %a = hip.add(%ctx) ins(%x, %v : tensor<1x3xi64>, tensor<1x3xi64>)
      outs(%e : tensor<1x3xi64>) -> tensor<1x3xi64>
  return %a : tensor<1x3xi64>
}

// -----

// A view chain with a host reader partway along: only the branch that reaches
// the hip op is cloned, and the host reader keeps the original view.
// CHECK-LABEL: func.func @mixed_view_chain
// CHECK-DAG:     %[[C:.*]] = arith.constant dense<7> : tensor<2x3xi32>
// CHECK-DAG:     %[[D:.*]] = hip.constant {value = dense<7> : tensor<2x3xi32>} : tensor<2x3xi32>
// CHECK-DAG:     %[[HV:.*]] = tensor.collapse_shape %[[C]] {{\[\[0, 1\]\]}}
// CHECK-DAG:     %[[DV:.*]] = tensor.collapse_shape %[[D]] {{\[\[0, 1\]\]}}
// CHECK-DAG:     tensor.extract %[[HV]]
// CHECK-DAG:     %[[DS:.*]] = tensor.extract_slice %[[DV]][2] [4] [1]
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %[[DS]] :
func.func @mixed_view_chain(%ctx: !hip.context, %x: tensor<4xi32>, %i: index)
    -> (i32, tensor<4xi32>) {
  %c = arith.constant dense<7> : tensor<2x3xi32>
  %flat = tensor.collapse_shape %c [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
  %s = tensor.extract %flat[%i] : tensor<6xi32>
  %tail = tensor.extract_slice %flat[2] [4] [1] : tensor<6xi32> to tensor<4xi32>
  %e = tensor.empty() : tensor<4xi32>
  %a = hip.add(%ctx) ins(%x, %tail : tensor<4xi32>, tensor<4xi32>)
      outs(%e : tensor<4xi32>) -> tensor<4xi32>
  return %s, %a : i32, tensor<4xi32>
}

// -----

// tensor.reshape is followed through its data operand only; its shape operand
// is read on the host and keeps the arith.constant. tensor.cast is a view too.
// CHECK-LABEL: func.func @through_reshape_and_cast
// CHECK-DAG:     %[[SHAPE:.*]] = arith.constant dense<2> : tensor<2xi64>
// CHECK-DAG:     %[[D:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<4xf32>} : tensor<4xf32>
// CHECK-DAG:     %[[DR:.*]] = tensor.reshape %[[D]](%[[SHAPE]])
// CHECK-DAG:     %[[DC:.*]] = tensor.cast %[[DR]] : tensor<2x2xf32> to tensor<?x2xf32>
// CHECK:         hip.cast(%{{.*}}) ins(%[[DC]] :
// CHECK-NOT:     hip.constant {value = dense<2> : tensor<2xi64>}
func.func @through_reshape_and_cast(%ctx: !hip.context, %n: index)
    -> tensor<?x2xf16> {
  %c = arith.constant dense<1.0> : tensor<4xf32>
  %shape = arith.constant dense<2> : tensor<2xi64>
  %r = tensor.reshape %c(%shape)
      : (tensor<4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
  %dyn = tensor.cast %r : tensor<2x2xf32> to tensor<?x2xf32>
  %e = tensor.empty(%n) : tensor<?x2xf16>
  %f = hip.cast(%ctx) ins(%dyn : tensor<?x2xf32>) outs(%e : tensor<?x2xf16>)
      {to = 10 : i64} : tensor<?x2xf16>
  return %f : tensor<?x2xf16>
}

// -----

// A DPS init is read from the device too: bufferization copies a constant into
// a fresh buffer before the op writes it, with a device-to-device memcpy.
// CHECK-LABEL: func.func @dps_init_promoted
// CHECK:         %[[D:.*]] = hip.constant {value = dense<0> : tensor<3xi64>} : tensor<3xi64>
// CHECK:         hip.add(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<3xi64>, tensor<3xi64>) outs(%[[D]] :
func.func @dps_init_promoted(%ctx: !hip.context, %x: tensor<3xi64>,
                             %y: tensor<3xi64>) -> tensor<3xi64> {
  %c = arith.constant dense<0> : tensor<3xi64>
  %a = hip.add(%ctx) ins(%x, %y : tensor<3xi64>, tensor<3xi64>)
      outs(%c : tensor<3xi64>) -> tensor<3xi64>
  return %a : tensor<3xi64>
}

// -----

// tensor.insert_slice copies its source into the destination's buffer with a
// device-to-device memcpy, so a constant source is a device read. This is how
// ONNX Concat lowers.
// CHECK-LABEL: func.func @insert_slice_source
// CHECK:         %[[D:.*]] = hip.constant {value = dense<0.000000e+00> : tensor<3x4xf32>} : tensor<3x4xf32>
// CHECK:         tensor.insert_slice %{{.*}} into
// CHECK:         tensor.insert_slice %[[D]] into
func.func @insert_slice_source(%x: tensor<2x4xf32>) -> tensor<5x4xf32> {
  %z = arith.constant dense<0.0> : tensor<3x4xf32>
  %e = tensor.empty() : tensor<5x4xf32>
  %0 = tensor.insert_slice %x into %e[0, 0] [2, 4] [1, 1]
      : tensor<2x4xf32> into tensor<5x4xf32>
  %1 = tensor.insert_slice %z into %0[2, 0] [3, 4] [1, 1]
      : tensor<3x4xf32> into tensor<5x4xf32>
  return %1 : tensor<5x4xf32>
}

// -----

// A constant destination is copied into a writable buffer before the slice is
// written, by the same memcpy.
// CHECK-LABEL: func.func @insert_slice_dest
// CHECK:         %[[D:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<4xf32>} : tensor<4xf32>
// CHECK:         tensor.insert_slice %{{.*}} into %[[D]][1] [2] [1]
func.func @insert_slice_dest(%x: tensor<2xf32>) -> tensor<4xf32> {
  %c = arith.constant dense<1.0> : tensor<4xf32>
  %0 = tensor.insert_slice %x into %c[1] [2] [1] : tensor<2xf32> into tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// hip.constant has no i4 form, so an i4 constant is left alone rather than
// turned into an op that fails verification.
// CHECK-LABEL: func.func @unsupported_element_type_stays
// CHECK-NOT:     hip.constant
// CHECK:         %[[C:.*]] = arith.constant dense<3> : tensor<4xi4>
// CHECK:         hip.cast(%{{.*}}) ins(%[[C]] :
func.func @unsupported_element_type_stays(%ctx: !hip.context) -> tensor<4xi8> {
  %c = arith.constant dense<3> : tensor<4xi4>
  %e = tensor.empty() : tensor<4xi8>
  %f = hip.cast(%ctx) ins(%c : tensor<4xi4>) outs(%e : tensor<4xi8>)
      {to = 3 : i64} : tensor<4xi8>
  return %f : tensor<4xi8>
}

// -----

// A readback of a hip.cast of a constant folds to a host constant, but only
// while the cast reads an arith.constant. The pass applies that fold before
// promoting, so the cast dies and nothing is promoted.
// CHECK-LABEL: func.func @readback_of_cast_folds_first
// CHECK-NOT:     hip.constant
// CHECK-NOT:     hip.readback_scalar
// CHECK:         %[[C:.*]] = arith.constant 7 : i32
// CHECK:         return %[[C]] : i32
func.func @readback_of_cast_folds_first(%ctx: !hip.context) -> i32 {
  %c = arith.constant dense<7> : tensor<i64>
  %init = tensor.empty() : tensor<i32>
  %gpu = hip.cast(%ctx) ins(%c : tensor<i64>) outs(%init : tensor<i32>)
      {to = 6 : i64} : tensor<i32>
  %host = hip.readback_scalar(%ctx, %gpu : tensor<i32>) -> i32
  return %host : i32
}
