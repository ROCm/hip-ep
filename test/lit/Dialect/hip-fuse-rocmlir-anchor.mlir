// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The anchor pass takes the modules of hip-fuse-rocmlir-elementwise whole into
// the anchor kernels. The second hip-mlir-opt re-verifies the outlined kernels.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s

// A prologue module that only the conv reads and an epilogue module that
// keeps the conv's shape both go into the conv kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[N:.*]] = hip.neg
// CHECK: %[[C:.*]] = hip.conv({{.*}}) ins(%[[N]],
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[C]],
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir_elementwise
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
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

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[MM:.*]] = hip.matmul
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[MM]] :
// CHECK: return %[[T]]
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} {{.*}} : tensor<8x4xf16>
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

// A module with no anchor that saves a launch becomes an elementwise kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: %[[A:.*]] = hip.add
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.add
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

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
// outside the conv kernel and becomes an elementwise kernel of its own.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: %[[A:.*]] = hip.add
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: return %[[M]]
// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[C:.*]] = hip.conv
// CHECK-NEXT: return %[[C]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir_elementwise
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: %[[E:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%[[K]],
// CHECK: return %[[E]]

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

// -----

// A module with two inputs is a prologue too: the 1x1 conv kernel loads both
// inputs and computes the add and the max on the fly.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.kernel}
// CHECK: %[[A:.*]] = hip.add
// CHECK: %[[MX:.*]] = hip.max({{.*}}) ins(%[[A]],
// CHECK: hip.conv({{.*}}) ins(%[[MX]],
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.add
// CHECK: hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%arg1, %arg2, %arg3, %arg4 :
// CHECK-NOT: hip.rocmlir

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %r: tensor<1x8x4x4xf16>, %w: tensor<16x8x1x1xf16>,
                      %b: tensor<16xf16>) -> tensor<1x16x4x4xf16> {
  %zero = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %ae = tensor.empty() : tensor<1x8x4x4xf16>
  %a = hip.add(%ctx) ins(%x, %r : tensor<1x8x4x4xf16>, tensor<1x8x4x4xf16>)
      outs(%ae : tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16>
  %me = tensor.empty() : tensor<1x8x4x4xf16>
  %m = hip.max(%ctx) ins(%a, %zero : tensor<1x8x4x4xf16>, tensor<f16>)
      outs(%me : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%m, %w, %b : tensor<1x8x4x4xf16>,
                                      tensor<16x8x1x1xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
       pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A 3x3 conv loads each input element nine times, so the two-input module
// stays its own kernel and computes the add and the max once.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: rock.elementwise
// CHECK: hip.add
// CHECK: hip.max
// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.kernel}
// CHECK-NOT: hip.add
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[M:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%arg1, %arg2 :
// CHECK: hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%[[M]], %arg3, %arg4 :

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x4xf16>,
                      %r: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                      %b: tensor<16xf16>) -> tensor<1x16x4x4xf16> {
  %zero = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %ae = tensor.empty() : tensor<1x8x4x4xf16>
  %a = hip.add(%ctx) ins(%x, %r : tensor<1x8x4x4xf16>, tensor<1x8x4x4xf16>)
      outs(%ae : tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16>
  %me = tensor.empty() : tensor<1x8x4x4xf16>
  %m = hip.max(%ctx) ins(%a, %zero : tensor<1x8x4x4xf16>, tensor<f16>)
      outs(%me : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%m, %w, %b : tensor<1x8x4x4xf16>,
                                      tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c : tensor<1x16x4x4xf16>
}

// -----

// A reshape without an init leads to a transpose: both are the epilogue.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[MM:.*]] = hip.matmul
// CHECK: %[[V:.*]] = tensor.expand_shape %[[MM]]
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[V]] :
// CHECK: return %[[T]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.expand_shape
// CHECK-NOT: hip.transpose

func.func @main_graph(%ctx: !hip.context, %x: tensor<4x16xf16>,
                      %y: tensor<16x8xf16>) -> tensor<2x4x4xf16> {
  %e = tensor.empty() : tensor<4x8xf16>
  %mm = hip.matmul(%ctx) ins(%x, %y : tensor<4x16xf16>, tensor<16x8xf16>)
      outs(%e : tensor<4x8xf16>) : tensor<4x8xf16>
  %v = tensor.expand_shape %mm [[0], [1, 2]] output_shape [4, 2, 4]
      : tensor<4x8xf16> into tensor<4x2x4xf16>
  %te = tensor.empty() : tensor<2x4x4xf16>
  %t = hip.transpose(%ctx) ins(%v : tensor<4x2x4xf16>)
      outs(%te : tensor<2x4x4xf16>) {perm = [1, 0, 2]} : tensor<2x4x4xf16>
  return %t : tensor<2x4x4xf16>
}

// -----

// A reshape without an init leads to a module: both are the epilogue.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[CC:.*]] = tensor.collapse_shape %[[C]]
// CHECK: %[[RC:.*]] = tensor.collapse_shape %arg3
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[RC]], %[[CC]] :
// CHECK: return %[[A]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.add
// CHECK-NOT: tensor.collapse_shape

func.func @main_graph(%ctx: !hip.context, %r: tensor<1x16x4x4xf16>,
                      %in: tensor<1x8x4x4xf16>, %w: tensor<16x8x3x3xf16>,
                      %b: tensor<16xf16>) -> tensor<1x16x16xf16> {
  %rc = tensor.collapse_shape %r [[0], [1], [2, 3]]
      : tensor<1x16x4x4xf16> into tensor<1x16x16xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %cc = tensor.collapse_shape %c [[0], [1], [2, 3]]
      : tensor<1x16x4x4xf16> into tensor<1x16x16xf16>
  %ae = tensor.empty() : tensor<1x16x16xf16>
  %a = hip.add(%ctx) ins(%rc, %cc : tensor<1x16x16xf16>, tensor<1x16x16xf16>)
      outs(%ae : tensor<1x16x16xf16>) -> tensor<1x16x16xf16>
  return %a : tensor<1x16x16xf16>
}
