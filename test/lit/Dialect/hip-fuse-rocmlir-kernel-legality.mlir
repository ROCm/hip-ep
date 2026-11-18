// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The rocMLIR fusion takes a pointwise op only when convert-hip-to-tosa
// lowers it inside the kernel (isTosaLowerableInKernel). A form with no TOSA
// spelling would stay a hip op in the rock.kernel, and a conv or gemm kernel
// then fails its compile; it stays a HIP op outside the kernel instead.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | FileCheck %s

// A float Mod with fmod = 1 has no TOSA spelling: it stays after the kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-NOT: hip.mod
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir
// CHECK: hip.mod(%{{.*}}) ins(%[[K]],
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf32>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> tensor<1x16x4x4xf32> {
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf32>,
                                        tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  %d = hip.constant {value = dense<3.0> : tensor<f32>} : tensor<f32>
  %me = tensor.empty() : tensor<1x16x4x4xf32>
  %m = hip.mod(%ctx) ins(%c, %d : tensor<1x16x4x4xf32>, tensor<f32>)
      outs(%me : tensor<1x16x4x4xf32>) {fmod = 1 : i64} : tensor<1x16x4x4xf32>
  return %m : tensor<1x16x4x4xf32>
}

// -----

// The same Mod as a lone prologue op stays before the kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-NOT: hip.mod
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[M:.*]] = hip.mod
// CHECK: hip.rocmlir(%{{.*}}) @{{.*}} ins(%[[M]],
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf32>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> tensor<1x16x4x4xf32> {
  %d = hip.constant {value = dense<3.0> : tensor<f32>} : tensor<f32>
  %me = tensor.empty() : tensor<1x8x4x4xf32>
  %m = hip.mod(%ctx) ins(%in, %d : tensor<1x8x4x4xf32>, tensor<f32>)
      outs(%me : tensor<1x8x4x4xf32>) {fmod = 1 : i64} : tensor<1x8x4x4xf32>
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%m, %w, %b : tensor<1x8x4x4xf32>,
                                       tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  return %c : tensor<1x16x4x4xf32>
}

// -----

// A cast from u8 lowers through unsigned_cast, so a u8 image normalized into
// a conv stays one kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: tensor<1x8x4x4xui8>
// CHECK: hip.cast
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.cast
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xui8>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>)
    -> tensor<1x16x4x4xf32> {
  %fe = tensor.empty() : tensor<1x8x4x4xf32>
  %f = hip.cast(%ctx) ins(%in : tensor<1x8x4x4xui8>)
      outs(%fe : tensor<1x8x4x4xf32>) {to = 1 : i64} : tensor<1x8x4x4xf32>
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%f, %w, %b : tensor<1x8x4x4xf32>,
                                       tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  return %c : tensor<1x16x4x4xf32>
}

// -----

// A Where on a u8 mask (ONNX BOOL) lowers inside the kernel and joins it.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: hip.conv
// CHECK: hip.where
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.where
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf32>,
                      %w: tensor<16x8x3x3xf32>, %b: tensor<16xf32>,
                      %mask: tensor<1x16x4x4xui8>, %x: tensor<f32>)
    -> tensor<1x16x4x4xf32> {
  %e = tensor.empty() : tensor<1x16x4x4xf32>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf32>,
                                        tensor<16x8x3x3xf32>, tensor<16xf32>)
      outs(%e : tensor<1x16x4x4xf32>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf32>
  %we = tensor.empty() : tensor<1x16x4x4xf32>
  %r = hip.where(%ctx) ins(%mask, %x, %c :
      tensor<1x16x4x4xui8>, tensor<f32>, tensor<1x16x4x4xf32>)
      outs(%we : tensor<1x16x4x4xf32>) : tensor<1x16x4x4xf32>
  return %r : tensor<1x16x4x4xf32>
}

// -----

// Outside any anchor, a u8 Clip (Max, then Min) outlines as an elementwise
// kernel of its own: HipToTosa computes both in i32, and the HIP path has no
// u8 Max or Min to fall back to.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: rock.nongemm
// CHECK: hip.max
// CHECK: hip.min
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.max
// CHECK: hip.rocmlir
// CHECK-NOT: hip.min
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x1x16x16xui8>)
    -> tensor<1x1x16x16xui8> {
  %lo = hip.constant {value = dense<100> : tensor<ui8>} : tensor<ui8>
  %hi = hip.constant {value = dense<200> : tensor<ui8>} : tensor<ui8>
  %me = tensor.empty() : tensor<1x1x16x16xui8>
  %m = hip.max(%ctx) ins(%x, %lo : tensor<1x1x16x16xui8>, tensor<ui8>)
      outs(%me : tensor<1x1x16x16xui8>) : tensor<1x1x16x16xui8>
  %ne = tensor.empty() : tensor<1x1x16x16xui8>
  %n = hip.min(%ctx) ins(%m, %hi : tensor<1x1x16x16xui8>, tensor<ui8>)
      outs(%ne : tensor<1x1x16x16xui8>) : tensor<1x1x16x16xui8>
  return %n : tensor<1x1x16x16xui8>
}
