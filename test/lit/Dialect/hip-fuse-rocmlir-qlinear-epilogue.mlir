// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Past the requantize of a quantized conv, hip-fuse-rocmlir takes the graph
// of dequantizes, pointwise ops and quantizes that reads the conv, and a
// dequantize that only that graph reads. Inside the kernel, a quantized add
// or mul of two dequantizes becomes the integer form that ONNX Runtime's
// QLinearAdd and QLinearMul compute, with a unit-scale quantize. A value that
// a dequantize widened is never a kernel result.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s | FileCheck %s

// Hard swish, y * q(q(y + 3) / 6): one kernel and one u8 result.
// The add is a * 0.25 / 0.5 + (0 - 0.5 * 128 + 3 * 1 / 0.5) = a * 0.5 - 58.
// The mul is (y - 128) * t * (0.25 * 0.125 / 0.0625) + 10.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x2x4x4xui8>
// CHECK: %[[Y:.*]] = hip.quantize_linear
// CHECK: %[[YF:.*]] = hip.cast(%{{.*}}) ins(%[[Y]] : {{.*}}) {to = 1 : i64}
// CHECK: %[[A:.*]] = hip.mul(%{{.*}}) ins(%[[YF]], %{{.*}} :
// CHECK: %[[B:.*]] = hip.constant {value = dense<-5.800000e+01> : tensor<f32>}
// CHECK: %[[S:.*]] = hip.add(%{{.*}}) ins(%[[A]], %[[B]] :
// CHECK-DAG: %[[ONE:.*]] = hip.constant {value = dense<1.000000e+00> : tensor<f32>}
// CHECK-DAG: %[[Z0:.*]] = hip.constant {value = dense<0> : tensor<ui8>}
// CHECK: %[[T:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[S]], %[[ONE]] : {{.*}}) zero_point(%[[Z0]] : tensor<ui8>)
// CHECK: hip.dequantize_linear(%{{.*}}) ins(%[[T]],
// CHECK: hip.reciprocal
// CHECK: %[[T2:.*]] = hip.quantize_linear
// CHECK: %[[T2F:.*]] = hip.cast(%{{.*}}) ins(%[[T2]] :
// CHECK: %[[YF2:.*]] = hip.cast(%{{.*}}) ins(%[[Y]] :
// CHECK: %[[ZA:.*]] = hip.constant {value = dense<1.280000e+02> : tensor<f32>}
// CHECK: %[[YC:.*]] = hip.sub(%{{.*}}) ins(%[[YF2]], %[[ZA]] :
// CHECK: %[[P:.*]] = hip.mul(%{{.*}}) ins(%[[YC]], %[[T2F]] :
// CHECK: %[[RATIO:.*]] = hip.constant {value = dense<5.000000e-01> : tensor<f32>}
// CHECK: %[[PS:.*]] = hip.mul(%{{.*}}) ins(%[[P]], %[[RATIO]] :
// CHECK: %[[ZC:.*]] = hip.constant {value = dense<1.000000e+01> : tensor<f32>}
// CHECK: %[[H:.*]] = hip.add(%{{.*}}) ins(%[[PS]], %[[ZC]] :
// CHECK: %[[OUT:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[H]],
// CHECK: return %[[OUT]] : tensor<1x2x4x4xui8>

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.dequantize_linear
// CHECK-NOT: hip.quantize_linear
// CHECK: hip.rocmlir
// CHECK-NOT: hip.dequantize_linear
// CHECK-NOT: hip.quantize_linear
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

  %fe0 = tensor.empty() : tensor<1x2x4x4xf32>
  %d0 = hip.dequantize_linear(%ctx) ins(%y, %ys : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%yz : tensor<ui8>) outs(%fe0 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %three = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %z0 = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %ce = tensor.empty() : tensor<f32>
  %c = hip.dequantize_linear(%ctx) ins(%three, %one : tensor<ui8>, tensor<f32>)
         zero_point(%z0 : tensor<ui8>) outs(%ce : tensor<f32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<f32>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %a = hip.add(%ctx) ins(%d0, %c : tensor<1x2x4x4xf32>, tensor<f32>)
         outs(%ae : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %ts = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %te = tensor.empty() : tensor<1x2x4x4xui8>
  %t = hip.quantize_linear(%ctx) ins(%a, %ts : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%z0 : tensor<ui8>) outs(%te : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>

  %fe1 = tensor.empty() : tensor<1x2x4x4xf32>
  %d1 = hip.dequantize_linear(%ctx) ins(%t, %ts : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z0 : tensor<ui8>) outs(%fe1 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %six = hip.constant {value = dense<6.0> : tensor<f32>} : tensor<f32>
  %re = tensor.empty() : tensor<f32>
  %r = hip.reciprocal(%ctx) ins(%six : tensor<f32>) outs(%re : tensor<f32>)
         : tensor<f32>
  %me = tensor.empty() : tensor<1x2x4x4xf32>
  %m = hip.mul(%ctx) ins(%d1, %r : tensor<1x2x4x4xf32>, tensor<f32>)
         outs(%me : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %t2s = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %t2e = tensor.empty() : tensor<1x2x4x4xui8>
  %t2 = hip.quantize_linear(%ctx) ins(%m, %t2s : tensor<1x2x4x4xf32>, tensor<f32>)
          zero_point(%z0 : tensor<ui8>) outs(%t2e : tensor<1x2x4x4xui8>)
          {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
           saturate = 1 : i64} : tensor<1x2x4x4xui8>

  %fe2 = tensor.empty() : tensor<1x2x4x4xf32>
  %d2 = hip.dequantize_linear(%ctx) ins(%y, %ys : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%yz : tensor<ui8>) outs(%fe2 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %fe3 = tensor.empty() : tensor<1x2x4x4xf32>
  %d3 = hip.dequantize_linear(%ctx) ins(%t2, %t2s : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%z0 : tensor<ui8>) outs(%fe3 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %he = tensor.empty() : tensor<1x2x4x4xf32>
  %h = hip.mul(%ctx) ins(%d2, %d3 : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
         outs(%he : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %zs = hip.constant {value = dense<0.0625> : tensor<f32>} : tensor<f32>
  %zz = hip.constant {value = dense<10> : tensor<ui8>} : tensor<ui8>
  %ze = tensor.empty() : tensor<1x2x4x4xui8>
  %z = hip.quantize_linear(%ctx) ins(%h, %zs : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%zz : tensor<ui8>) outs(%ze : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  return %z : tensor<1x2x4x4xui8>
}

// -----

// Residual add: the dequantize of the other side is read only by the add, so
// it joins and the kernel reads the u8 residual. With rA = 0.25 / 0.5 and
// rB = 0.125 / 0.5, the sum is a * 0.5 + (r * 0.25 + (3 - (0.5 * 128 +
// 0.25 * 4))) = a * 0.5 + (r * 0.25 - 62).

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: %[[R:[^:]*]]: tensor<1x2x4x4xui8>) -> tensor<1x2x4x4xui8>
// CHECK: %[[Y:.*]] = hip.quantize_linear
// CHECK: %[[YF:.*]] = hip.cast(%{{.*}}) ins(%[[Y]] :
// CHECK: %[[A:.*]] = hip.mul(%{{.*}}) ins(%[[YF]], %{{.*}} :
// CHECK: %[[RF:.*]] = hip.cast(%{{.*}}) ins(%[[R]] :
// CHECK: %[[RS:.*]] = hip.mul(%{{.*}}) ins(%[[RF]], %{{.*}} :
// CHECK: %[[FIXED:.*]] = hip.constant {value = dense<-6.200000e+01> : tensor<f32>}
// CHECK: %[[BP:.*]] = hip.add(%{{.*}}) ins(%[[RS]], %[[FIXED]] :
// CHECK: %[[S:.*]] = hip.add(%{{.*}}) ins(%[[A]], %[[BP]] :
// CHECK: %[[OUT:.*]] = hip.quantize_linear(%{{.*}}) ins(%[[S]],
// CHECK-NOT: hip.dequantize_linear
// CHECK: return %[[OUT]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.dequantize_linear
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>,
                      %res: tensor<1x2x4x4xui8>) -> tensor<1x2x4x4xui8> {
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
  %fe0 = tensor.empty() : tensor<1x2x4x4xf32>
  %d0 = hip.dequantize_linear(%ctx) ins(%y, %ys : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%yz : tensor<ui8>) outs(%fe0 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %rs = hip.constant {value = dense<0.125> : tensor<f32>} : tensor<f32>
  %rz = hip.constant {value = dense<4> : tensor<ui8>} : tensor<ui8>
  %fe1 = tensor.empty() : tensor<1x2x4x4xf32>
  %d1 = hip.dequantize_linear(%ctx) ins(%res, %rs : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%rz : tensor<ui8>) outs(%fe1 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %ae = tensor.empty() : tensor<1x2x4x4xf32>
  %a = hip.add(%ctx) ins(%d0, %d1 : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
         outs(%ae : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %os = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %oz = hip.constant {value = dense<3> : tensor<ui8>} : tensor<ui8>
  %oe = tensor.empty() : tensor<1x2x4x4xui8>
  %o = hip.quantize_linear(%ctx) ins(%a, %os : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%oz : tensor<ui8>) outs(%oe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  return %o : tensor<1x2x4x4xui8>
}

// -----

// A dequantize whose f32 value leaves the graph stays outside: the kernel
// would otherwise write four bytes for each one it reads.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x2x4x4xui8>
// CHECK-NOT: hip.dequantize_linear
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[Y:.*]] = hip.rocmlir
// CHECK: hip.dequantize_linear(%{{.*}}) ins(%[[Y]],
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x4x4xui8>)
    -> tensor<1x2x4x4xf32> {
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
  %fe = tensor.empty() : tensor<1x2x4x4xf32>
  %d = hip.dequantize_linear(%ctx) ins(%y, %ys : tensor<1x2x4x4xui8>, tensor<f32>)
         zero_point(%yz : tensor<ui8>) outs(%fe : tensor<1x2x4x4xf32>)
         {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  return %d : tensor<1x2x4x4xf32>
}

// -----

// Squeeze-and-excitation: the conv gives one value per channel, and the mul
// broadcasts it over the plane. That would make the kernel compute a larger
// output than its GEMM, so the mul and its dequantizes stay outside.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: -> tensor<1x2x1x1xui8>
// CHECK-NOT: hip.dequantize_linear
// CHECK: return

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.dequantize_linear
// CHECK: hip.dequantize_linear
// CHECK: hip.mul
// CHECK: hip.quantize_linear
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x1x1xui8>,
                      %plane: tensor<1x2x4x4xui8>) -> tensor<1x2x4x4xui8> {
  %xs = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %xz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %w = hip.constant {value = dense<1> : tensor<2x2x1x1xi8>} : tensor<2x2x1x1xi8>
  %ws = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %wz = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ys = hip.constant {value = dense<0.25> : tensor<f32>} : tensor<f32>
  %yz = hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  %ye = tensor.empty() : tensor<1x2x1x1xui8>
  %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz :
         tensor<1x2x1x1xui8>, tensor<f32>, tensor<ui8>, tensor<2x2x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>)
         outs(%ye : tensor<1x2x1x1xui8>)
         {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
          pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x2x1x1xui8>
  %fe0 = tensor.empty() : tensor<1x2x1x1xf32>
  %d0 = hip.dequantize_linear(%ctx) ins(%y, %ys : tensor<1x2x1x1xui8>, tensor<f32>)
          zero_point(%yz : tensor<ui8>) outs(%fe0 : tensor<1x2x1x1xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x1x1xf32>
  %fe1 = tensor.empty() : tensor<1x2x4x4xf32>
  %d1 = hip.dequantize_linear(%ctx) ins(%plane, %ys : tensor<1x2x4x4xui8>, tensor<f32>)
          zero_point(%yz : tensor<ui8>) outs(%fe1 : tensor<1x2x4x4xf32>)
          {axis = 1 : i64, block_size = 0 : i64} : tensor<1x2x4x4xf32>
  %me = tensor.empty() : tensor<1x2x4x4xf32>
  %m = hip.mul(%ctx) ins(%d1, %d0 : tensor<1x2x4x4xf32>, tensor<1x2x1x1xf32>)
         outs(%me : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %os = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %oe = tensor.empty() : tensor<1x2x4x4xui8>
  %o = hip.quantize_linear(%ctx) ins(%m, %os : tensor<1x2x4x4xf32>, tensor<f32>)
         zero_point(%yz : tensor<ui8>) outs(%oe : tensor<1x2x4x4xui8>)
         {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
          saturate = 1 : i64} : tensor<1x2x4x4xui8>
  return %o : tensor<1x2x4x4xui8>
}
