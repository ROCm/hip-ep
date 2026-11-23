// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// rocMLIR takes only signless integers, so HipToTosa relabels an unsigned
// kernel to signless, and it can do that only when each op keeps an unsigned
// value's meaning. A conv or gemm kernel that rocMLIR rejects fails the
// compile, so the anchor fusion takes an unsigned op only when it is one of
// these: a ui8 Max, Min, Clip (Max and Min) or Less, which HipToTosa computes
// in i32, joins the kernel; a ui8 Add stays a HIP op outside it.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[Y:.*]] = hip.quantize_linear
// CHECK: %[[M:.*]] = hip.max(%{{.*}}) ins(%[[Y]],
// CHECK: return %[[M]] : tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.max
// CHECK: hip.rocmlir
// CHECK-NOT: hip.max
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%ye : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xui8>
  %lo = hip.constant {value = dense<200> : tensor<ui8>} : tensor<ui8>
  %me = tensor.empty() : tensor<1x2x4x4xui8>
  %m = hip.max(%ctx) ins(%y, %lo : tensor<1x2x4x4xui8>, tensor<ui8>)
         outs(%me : tensor<1x2x4x4xui8>) : tensor<1x2x4x4xui8>
  return %m : tensor<1x2x4x4xui8>
}

// -----

// Clip lowers to a Max and a Min; both join the kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[Y:.*]] = hip.quantize_linear
// CHECK: %[[M:.*]] = hip.max(%{{.*}}) ins(%[[Y]],
// CHECK: %[[N:.*]] = hip.min(%{{.*}}) ins(%[[M]],
// CHECK: return %[[N]] : tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.m{{ax|in}}
// CHECK: hip.rocmlir
// CHECK-NOT: hip.m{{ax|in}}
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%ye : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xui8>
  %lo = hip.constant {value = dense<100> : tensor<ui8>} : tensor<ui8>
  %hi = hip.constant {value = dense<200> : tensor<ui8>} : tensor<ui8>
  %me = tensor.empty() : tensor<1x2x4x4xui8>
  %m = hip.max(%ctx) ins(%y, %lo : tensor<1x2x4x4xui8>, tensor<ui8>)
         outs(%me : tensor<1x2x4x4xui8>) : tensor<1x2x4x4xui8>
  %ne = tensor.empty() : tensor<1x2x4x4xui8>
  %n = hip.min(%ctx) ins(%m, %hi : tensor<1x2x4x4xui8>, tensor<ui8>)
         outs(%ne : tensor<1x2x4x4xui8>) : tensor<1x2x4x4xui8>
  return %n : tensor<1x2x4x4xui8>
}

// -----

// A ui8 Less with an i1 result joins the kernel too: HipToTosa compares in
// i32, so 200 < 100 stays false. (An ONNX Less returns BOOL, which is ui8 in
// hip; that form has no TOSA spelling and runs after the kernel.)

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[Y:.*]] = hip.quantize_linear
// CHECK: %[[L:.*]] = hip.less(%{{.*}}) ins(%[[Y]],
// CHECK: return %[[L]] : tensor<1x2x4x4xi1>
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.less
// CHECK: hip.rocmlir
// CHECK-NOT: hip.less
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xi1> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%ye : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xui8>
  %t = hip.constant {value = dense<100> : tensor<ui8>} : tensor<ui8>
  %le = tensor.empty() : tensor<1x2x4x4xi1>
  %l = hip.less(%ctx) ins(%y, %t : tensor<1x2x4x4xui8>, tensor<ui8>)
         outs(%le : tensor<1x2x4x4xi1>) : tensor<1x2x4x4xi1>
  return %l : tensor<1x2x4x4xi1>
}

// -----

// A ui8 Add has no form that keeps its meaning under the signless relabel.
// It stays outside the kernel, and the kernel returns the u8 conv result.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-NOT: hip.add{{.*}}xui8>
// CHECK: return %{{.*}} : tensor<1x2x4x4xui8>
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir
// CHECK: hip.add(%{{.*}}) ins(%[[K]],
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x4x4xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%ye : tensor<1x2x4x4xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x4x4xui8>
  %t = hip.constant {value = dense<100> : tensor<ui8>} : tensor<ui8>
  %ae = tensor.empty() : tensor<1x2x4x4xui8>
  %a = hip.add(%ctx) ins(%y, %t : tensor<1x2x4x4xui8>, tensor<ui8>)
         outs(%ae : tensor<1x2x4x4xui8>) -> tensor<1x2x4x4xui8>
  return %a : tensor<1x2x4x4xui8>
}
