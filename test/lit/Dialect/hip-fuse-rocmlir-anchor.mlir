// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The anchor pass takes the modules of hip-fuse-rocmlir-elementwise whole into
// the anchor kernels. The second hip-mlir-opt re-verifies the outlined kernels.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// A prologue module that only the conv reads and an epilogue module that
// keeps the conv's shape both go into the conv kernel.

// CHECK-LABEL: func.func @rocMlir0
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[N:.*]] = hip.neg
// CHECK: %[[C:.*]] = hip.conv({{.*}}) ins(%[[N]],
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[C]],
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir_elementwise
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @rocMlir0
// CHECK-NOT: hip.neg
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
  %ne = tensor.empty() : tensor<1x8x4x4xf16>
  %n = hip.neg(%ctx) ins(%in : tensor<1x8x4x4xf16>)
      outs(%ne : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%n, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%c, %r : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// A lone transpose that reads the matmul is the kernel's epilogue.

// CHECK-LABEL: func.func @rocMlir0
// CHECK: %[[MM:.*]] = hip.matmul
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[MM]] :
// CHECK: return %[[T]]
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @rocMlir0 {{.*}} : tensor<8x4xf16>
// CHECK-NOT: hip.transpose
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<4x16xf16>,
                      %y: tensor<16x8xf16>) -> tensor<8x4xf16> {
  %e = tensor.empty() : tensor<4x8xf16>
  %mm = hip.matmul(%ctx) ins(%x, %y : tensor<4x16xf16>, tensor<16x8xf16>)
      outs(%e : tensor<4x8xf16>) : tensor<4x8xf16>
  %te = tensor.empty() : tensor<8x4xf16>
  %t = hip.transpose(%ctx) ins(%mm : tensor<4x8xf16>)
      outs(%te : tensor<8x4xf16>) {perm = [1, 0]} : tensor<8x4xf16>
  return %t : tensor<8x4xf16>
}

// -----

// A module with no anchor is inlined back into main_graph.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: %[[A:.*]] = hip.add
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<4x16xf16>,
                      %y: tensor<4x16xf16>) -> tensor<4x16xf16> {
  %ae = tensor.empty() : tensor<4x16xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%ae : tensor<4x16xf16>) -> tensor<4x16xf16>
  %me = tensor.empty() : tensor<4x16xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%me : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %m : tensor<4x16xf16>
}

// -----

// An epilogue module that broadcasts the conv result to a larger shape stays
// outside the kernel, and its ops are inlined back.

// CHECK-LABEL: func.func @rocMlir0
// CHECK: %[[C:.*]] = hip.conv
// CHECK-NEXT: return %[[C]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir_elementwise
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @rocMlir0
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[K]],
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]

func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %z: tensor<2x16x4x4xf16>) -> tensor<2x16x4x4xf16> {
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<2x16x4x4xf16>
  %a = hip.add(%ctx) ins(%c, %z : tensor<1x16x4x4xf16>, tensor<2x16x4x4xf16>)
      outs(%ae : tensor<2x16x4x4xf16>) -> tensor<2x16x4x4xf16>
  %me = tensor.empty() : tensor<2x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<2x16x4x4xf16>, tensor<2x16x4x4xf16>)
      outs(%me : tensor<2x16x4x4xf16>) -> tensor<2x16x4x4xf16>
  return %m : tensor<2x16x4x4xf16>
}
