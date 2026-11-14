// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// With int4-weights, a 1x1 convolution takes the packed-int4 dequantize of
// its weights into its kernel, which then reads the 4-bit weights. The zero
// point is dequantized outside the kernel. Without the option, the dequantize
// stays a HIP op and the kernel reads the f32 weights.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor="int4-weights=true" %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | FileCheck %s --check-prefix=OFF

// Per-channel scale and zero point. The kernel computes
// (dequantize(w, 1) - zp) * scale on the packed weights; the zero point is
// dequantized once outside.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: %[[W:[a-z0-9]+]]: tensor<32x16x1x1xi8>
// CHECK-SAME: %[[ZP:[a-z0-9]+]]: tensor<32xf32>
// CHECK: hip.dequantize_linear({{.*}}) ins(%[[W]],
// CHECK-SAME: packed_int4
// CHECK: hip.sub
// CHECK: hip.mul
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<32xi8>,
// CHECK-SAME: packed_int4
// CHECK: hip.rocmlir
// CHECK-NOT: hip.dequantize_linear

// OFF-LABEL: func.func @main_graph
// OFF: hip.dequantize_linear
// OFF-SAME: outs(%{{.*}} : tensor<32x16x1x1xf32>)
// OFF: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x1x8xf32>,
                      %w: tensor<32x16x1x1xi8>, %zp: tensor<32xi8>)
    -> tensor<1x32x1x8xf32> {
  %s = hip.constant {value = dense<5.000000e-01> : tensor<32xf32>}
      : tensor<32xf32>
  %we = tensor.empty() : tensor<32x16x1x1xf32>
  %wd = hip.dequantize_linear(%ctx) ins(%w, %s : tensor<32x16x1x1xi8>,
                                                 tensor<32xf32>)
      zero_point(%zp : tensor<32xi8>) outs(%we : tensor<32x16x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4}
      : tensor<32x16x1x1xf32>
  %e = tensor.empty() : tensor<1x32x1x8xf32>
  %c = hip.conv(%ctx) ins(%x, %wd : tensor<1x16x1x8xf32>,
                                    tensor<32x16x1x1xf32>)
      outs(%e : tensor<1x32x1x8xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x32x1x8xf32>
  return %c : tensor<1x32x1x8xf32>
}

// -----

// Without a zero point the kernel only scales the packed weights.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: hip.dequantize_linear
// CHECK-SAME: packed_int4
// CHECK-NOT: hip.sub
// CHECK: hip.mul
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.dequantize_linear
// CHECK: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x1x8xf32>,
                      %w: tensor<32x16x1x1xi8>) -> tensor<1x32x1x8xf32> {
  %s = hip.constant {value = dense<5.000000e-01> : tensor<32xf32>}
      : tensor<32xf32>
  %we = tensor.empty() : tensor<32x16x1x1xf32>
  %wd = hip.dequantize_linear(%ctx) ins(%w, %s : tensor<32x16x1x1xi8>,
                                                 tensor<32xf32>)
      outs(%we : tensor<32x16x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4}
      : tensor<32x16x1x1xf32>
  %e = tensor.empty() : tensor<1x32x1x8xf32>
  %c = hip.conv(%ctx) ins(%x, %wd : tensor<1x16x1x8xf32>,
                                    tensor<32x16x1x1xf32>)
      outs(%e : tensor<1x32x1x8xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x32x1x8xf32>
  return %c : tensor<1x32x1x8xf32>
}

// -----

// A 3x3 filter is not a 1x1 convolution: the dequantize stays outside.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-NOT: packed_int4
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK-SAME: packed_int4
// CHECK: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x4x4xf32>,
                      %w: tensor<32x16x3x3xi8>) -> tensor<1x32x4x4xf32> {
  %s = hip.constant {value = dense<5.000000e-01> : tensor<32xf32>}
      : tensor<32xf32>
  %we = tensor.empty() : tensor<32x16x3x3xf32>
  %wd = hip.dequantize_linear(%ctx) ins(%w, %s : tensor<32x16x3x3xi8>,
                                                 tensor<32xf32>)
      outs(%we : tensor<32x16x3x3xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4}
      : tensor<32x16x3x3xf32>
  %e = tensor.empty() : tensor<1x32x4x4xf32>
  %c = hip.conv(%ctx) ins(%x, %wd : tensor<1x16x4x4xf32>,
                                    tensor<32x16x3x3xf32>)
      outs(%e : tensor<1x32x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x32x4x4xf32>
  return %c : tensor<1x32x4x4xf32>
}

// -----

// An odd input-channel count would split a byte between two filter rows.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK-SAME: packed_int4
// CHECK: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x15x1x8xf32>,
                      %w: tensor<32x15x1x1xi8>) -> tensor<1x32x1x8xf32> {
  %s = hip.constant {value = dense<5.000000e-01> : tensor<32xf32>}
      : tensor<32xf32>
  %we = tensor.empty() : tensor<32x15x1x1xf32>
  %wd = hip.dequantize_linear(%ctx) ins(%w, %s : tensor<32x15x1x1xi8>,
                                                 tensor<32xf32>)
      outs(%we : tensor<32x15x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4}
      : tensor<32x15x1x1xf32>
  %e = tensor.empty() : tensor<1x32x1x8xf32>
  %c = hip.conv(%ctx) ins(%x, %wd : tensor<1x15x1x8xf32>,
                                    tensor<32x15x1x1xf32>)
      outs(%e : tensor<1x32x1x8xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x32x1x8xf32>
  return %c : tensor<1x32x1x8xf32>
}

// -----

// Unsigned int4 (ui8 storage) needs an unsigned conversion that this path
// does not emit, so its dequantize stays outside.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.dequantize_linear
// CHECK-SAME: packed_int4
// CHECK: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x1x8xf32>,
                      %w: tensor<32x16x1x1xui8>) -> tensor<1x32x1x8xf32> {
  %s = hip.constant {value = dense<5.000000e-01> : tensor<32xf32>}
      : tensor<32xf32>
  %we = tensor.empty() : tensor<32x16x1x1xf32>
  %wd = hip.dequantize_linear(%ctx) ins(%w, %s : tensor<32x16x1x1xui8>,
                                                 tensor<32xf32>)
      outs(%we : tensor<32x16x1x1xf32>)
      {axis = 0 : i64, block_size = 0 : i64, packed_int4}
      : tensor<32x16x1x1xf32>
  %e = tensor.empty() : tensor<1x32x1x8xf32>
  %c = hip.conv(%ctx) ins(%x, %wd : tensor<1x16x1x8xf32>,
                                    tensor<32x16x1x1xf32>)
      outs(%e : tensor<1x32x1x8xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x32x1x8xf32>
  return %c : tensor<1x32x1x8xf32>
}
