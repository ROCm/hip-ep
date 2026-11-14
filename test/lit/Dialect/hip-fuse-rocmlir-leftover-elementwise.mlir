// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// A pointwise module that no anchor kernel takes becomes a rock.elementwise
// kernel when it saves a launch. The second hip-mlir-opt re-parses and
// re-verifies each module.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-fusion %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s
// RUN: hip-mlir-opt --split-input-file \
// RUN:   --hip-rocmlir-fusion="standalone-elementwise=false" %s \
// RUN:   | FileCheck %s --check-prefix=NOEW

// A chain with no anchor is one kernel. Its inline scalar constant is cloned
// into the kernel, not passed as an argument. With standalone-elementwise=false
// no module becomes a kernel of its own, in this case or any later one.

// NOEW-NOT: rock.elementwise
// NOEW-LABEL: func.func @main_graph
// NOEW-NOT: hip.rocmlir
// NOEW: hip.add
// NOEW: hip.max
// NOEW: hip.neg
// NOEW-NOT: rock.elementwise

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%[[X:.*]]: tensor<4x8xf16>, %[[Y:.*]]: tensor<4x8xf16>) -> tensor<4x8xf16>
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: %[[ZERO:.*]] = hip.constant
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[X]], %[[Y]] :
// CHECK: %[[M:.*]] = hip.max({{.*}}) ins(%[[A]], %[[ZERO]] :
// CHECK: %[[N:.*]] = hip.neg({{.*}}) ins(%[[M]] :
// CHECK: return %[[N]]
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.add
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<4x8xf16>,
                      %y: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %zero = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %ae = tensor.empty() : tensor<4x8xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<4x8xf16>, tensor<4x8xf16>)
      outs(%ae : tensor<4x8xf16>) -> tensor<4x8xf16>
  %me = tensor.empty() : tensor<4x8xf16>
  %m = hip.max(%ctx) ins(%a, %zero : tensor<4x8xf16>, tensor<f16>)
      outs(%me : tensor<4x8xf16>) : tensor<4x8xf16>
  %ne = tensor.empty() : tensor<4x8xf16>
  %n = hip.neg(%ctx) ins(%m : tensor<4x8xf16>)
      outs(%ne : tensor<4x8xf16>) : tensor<4x8xf16>
  return %n : tensor<4x8xf16>
}

// -----

// One op saves no launch, so it stays a HIP op.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.neg

func.func @main_graph(%ctx: !hip.context, %x: tensor<4x8xf16>)
    -> tensor<4x8xf16> {
  %ne = tensor.empty() : tensor<4x8xf16>
  %n = hip.neg(%ctx) ins(%x : tensor<4x8xf16>)
      outs(%ne : tensor<4x8xf16>) : tensor<4x8xf16>
  return %n : tensor<4x8xf16>
}

// -----

// A group that the conv cannot take as its prologue, because the graph also
// returns its result, becomes its own kernel after the conv kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: %[[N:.*]] = hip.neg
// CHECK: %[[S:.*]] = hip.sigmoid({{.*}}) ins(%[[N]] :
// CHECK: return %[[S]]
// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.kernel}
// CHECK: hip.conv
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[P:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: %[[C:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%[[P]],
// CHECK: return %[[C]], %[[P]]

func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>)
    -> (tensor<1x16x4x4xf16>, tensor<1x8x4x4xf16>) {
  %ne = tensor.empty() : tensor<1x8x4x4xf16>
  %n = hip.neg(%ctx) ins(%in : tensor<1x8x4x4xf16>)
      outs(%ne : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
  %se = tensor.empty() : tensor<1x8x4x4xf16>
  %s = hip.sigmoid(%ctx) ins(%n : tensor<1x8x4x4xf16>)
      outs(%se : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%s, %w, %b : tensor<1x8x4x4xf16>,
                                       tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  return %c, %s : tensor<1x16x4x4xf16>, tensor<1x8x4x4xf16>
}

// -----

// rocMLIR cannot store a result that only transposes a kernel argument next
// to another result, so this module stays on the host.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.mul
// CHECK: hip.transpose

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x8x3xf32>,
                      %y: tensor<1x8x8x3xf32>)
    -> (tensor<1x8x8x3xf32>, tensor<1x3x8x8xf32>) {
  %me = tensor.empty() : tensor<1x8x8x3xf32>
  %m = hip.mul(%ctx) ins(%x, %y : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>)
      outs(%me : tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %te = tensor.empty() : tensor<1x3x8x8xf32>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x8x8x3xf32>)
      outs(%te : tensor<1x3x8x8xf32>) {perm = [0, 3, 1, 2]} : tensor<1x3x8x8xf32>
  return %m, %t : tensor<1x8x8x3xf32>, tensor<1x3x8x8xf32>
}

// -----

// A result transpose that moves the innermost dimension leaves the kernel:
// the kernel stores the value in the layout it computes in, and a HIP
// transpose follows it.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: -> tensor<1x4x32x32xf16>
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK-NOT: hip.transpose
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[K]] :
// CHECK: return %[[T]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x32x32xf16>,
                      %y: tensor<1x4x32x32xf16>) -> tensor<1x32x4x32xf16> {
  %ae = tensor.empty() : tensor<1x4x32x32xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x4x32x32xf16>, tensor<1x4x32x32xf16>)
      outs(%ae : tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
  %me = tensor.empty() : tensor<1x4x32x32xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x4x32x32xf16>, tensor<1x4x32x32xf16>)
      outs(%me : tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
  %te = tensor.empty() : tensor<1x32x4x32xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x4x32x32xf16>)
      outs(%te : tensor<1x32x4x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x32x4x32xf16>
  return %t : tensor<1x32x4x32xf16>
}

// -----

// When the graph also returns the untransposed value, the kernel has that one
// result, and the HIP transpose reads it.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: -> tensor<1x4x32x32xf16>
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK-NOT: hip.transpose
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[K]] :
// CHECK: return %[[K]], %[[T]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x32x32xf16>,
                      %y: tensor<1x4x32x32xf16>)
    -> (tensor<1x4x32x32xf16>, tensor<1x32x4x32xf16>) {
  %ae = tensor.empty() : tensor<1x4x32x32xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x4x32x32xf16>, tensor<1x4x32x32xf16>)
      outs(%ae : tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
  %me = tensor.empty() : tensor<1x4x32x32xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x4x32x32xf16>, tensor<1x4x32x32xf16>)
      outs(%me : tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
  %te = tensor.empty() : tensor<1x32x4x32xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x4x32x32xf16>)
      outs(%te : tensor<1x32x4x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x32x4x32xf16>
  return %m, %t : tensor<1x4x32x32xf16>, tensor<1x32x4x32xf16>
}

// -----

// An input transpose that moves the innermost dimension runs before the
// kernel, which reads the transposed tensor.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%{{.*}}: tensor<1x32x4x32xf16>, %{{.*}}: tensor<1x32x4x32xf16>) -> tensor<1x32x4x32xf16>
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK-NOT: hip.transpose
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%{{.*}} : tensor<1x4x32x32xf16>)
// CHECK: hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%[[T]],

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x32x32xf16>,
                      %y: tensor<1x32x4x32xf16>) -> tensor<1x32x4x32xf16> {
  %te = tensor.empty() : tensor<1x32x4x32xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<1x4x32x32xf16>)
      outs(%te : tensor<1x32x4x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x32x4x32xf16>
  %ae = tensor.empty() : tensor<1x32x4x32xf16>
  %a = hip.add(%ctx) ins(%t, %y : tensor<1x32x4x32xf16>, tensor<1x32x4x32xf16>)
      outs(%ae : tensor<1x32x4x32xf16>) -> tensor<1x32x4x32xf16>
  %me = tensor.empty() : tensor<1x32x4x32xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x32x4x32xf16>, tensor<1x32x4x32xf16>)
      outs(%me : tensor<1x32x4x32xf16>) -> tensor<1x32x4x32xf16>
  return %m : tensor<1x32x4x32xf16>
}

// -----

// An input innermost dimension of 3 elements, under peel-min-input-elements,
// gives a stride that still uses several elements of each cache line, so the
// transpose stays.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: hip.transpose
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x8x3xf16>,
                      %y: tensor<1x8x8x3xf16>) -> tensor<1x3x8x8xf16> {
  %ae = tensor.empty() : tensor<1x8x8x3xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x8x8x3xf16>, tensor<1x8x8x3xf16>)
      outs(%ae : tensor<1x8x8x3xf16>) -> tensor<1x8x8x3xf16>
  %me = tensor.empty() : tensor<1x8x8x3xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x8x8x3xf16>, tensor<1x8x8x3xf16>)
      outs(%me : tensor<1x8x8x3xf16>) -> tensor<1x8x8x3xf16>
  %te = tensor.empty() : tensor<1x3x8x8xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x8x8x3xf16>)
      outs(%te : tensor<1x3x8x8xf16>) {perm = [0, 3, 1, 2]} : tensor<1x3x8x8xf16>
  return %t : tensor<1x3x8x8xf16>
}

// -----

// With an output innermost dimension of 2, under peel-min-output-elements, the
// tile reads two contiguous input planes, so the transpose stays.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: hip.transpose
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x2x16x16xf16>,
                      %y: tensor<1x2x16x16xf16>) -> tensor<1x16x16x2xf16> {
  %ae = tensor.empty() : tensor<1x2x16x16xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16>)
      outs(%ae : tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16>
  %me = tensor.empty() : tensor<1x2x16x16xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x2x16x16xf16>, tensor<1x2x16x16xf16>)
      outs(%me : tensor<1x2x16x16xf16>) -> tensor<1x2x16x16xf16>
  %te = tensor.empty() : tensor<1x16x16x2xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x2x16x16xf16>)
      outs(%te : tensor<1x16x16x2xf16>) {perm = [0, 2, 3, 1]} : tensor<1x16x16x2xf16>
  return %t : tensor<1x16x16x2xf16>
}

// -----

// An output innermost dimension of 16 elements is under
// peel-min-output-elements, so the transpose stays.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: hip.transpose
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x32xf32>,
                      %y: tensor<1x16x32xf32>) -> tensor<1x32x16xf32> {
  %ae = tensor.empty() : tensor<1x16x32xf32>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x16x32xf32>, tensor<1x16x32xf32>)
      outs(%ae : tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  %me = tensor.empty() : tensor<1x16x32xf32>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x16x32xf32>, tensor<1x16x32xf32>)
      outs(%me : tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  %te = tensor.empty() : tensor<1x32x16xf32>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x16x32xf32>)
      outs(%te : tensor<1x32x16xf32>) {perm = [0, 2, 1]} : tensor<1x32x16xf32>
  return %t : tensor<1x32x16xf32>
}

// -----

// A transpose that keeps the innermost dimension in place stays in the kernel.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: hip.transpose
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.transpose
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x4x16xf16>,
                      %y: tensor<1x8x4x16xf16>) -> tensor<1x4x8x16xf16> {
  %ae = tensor.empty() : tensor<1x8x4x16xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x8x4x16xf16>, tensor<1x8x4x16xf16>)
      outs(%ae : tensor<1x8x4x16xf16>) -> tensor<1x8x4x16xf16>
  %me = tensor.empty() : tensor<1x8x4x16xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<1x8x4x16xf16>, tensor<1x8x4x16xf16>)
      outs(%me : tensor<1x8x4x16xf16>) -> tensor<1x8x4x16xf16>
  %te = tensor.empty() : tensor<1x4x8x16xf16>
  %t = hip.transpose(%ctx) ins(%m : tensor<1x8x4x16xf16>)
      outs(%te : tensor<1x4x8x16xf16>) {perm = [0, 2, 1, 3]} : tensor<1x4x8x16xf16>
  return %t : tensor<1x4x8x16xf16>
}

// -----

// After the peel, one op is left; it saves no launch, so both ops stay HIP ops.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.neg
// CHECK: hip.transpose

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x4x32x32xf16>)
    -> tensor<1x32x4x32xf16> {
  %ne = tensor.empty() : tensor<1x4x32x32xf16>
  %n = hip.neg(%ctx) ins(%x : tensor<1x4x32x32xf16>)
      outs(%ne : tensor<1x4x32x32xf16>) : tensor<1x4x32x32xf16>
  %te = tensor.empty() : tensor<1x32x4x32xf16>
  %t = hip.transpose(%ctx) ins(%n : tensor<1x4x32x32xf16>)
      outs(%te : tensor<1x32x4x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x32x4x32xf16>
  return %t : tensor<1x32x4x32xf16>
}

// -----

// The transpose reads a reshape that another module op also reads. It leaves
// the kernel with its own copy of the reshape, and the kernel keeps the other.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: -> (tensor<1x128x32xf16>, tensor<1x4x32x32xf16>)
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK-NOT: hip.transpose
// CHECK: return
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[K:.*]]:2 = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}}
// CHECK: %[[E:.*]] = tensor.expand_shape %[[K]]#0
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[E]] :
// CHECK: return %[[T]], %[[K]]#1

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                      %y: tensor<1x128x32xf16>, %z: tensor<1x4x32x32xf16>)
    -> (tensor<1x32x4x32xf16>, tensor<1x4x32x32xf16>) {
  %ae = tensor.empty() : tensor<1x128x32xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x128x32xf16>)
      outs(%ae : tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
  %e = tensor.expand_shape %a [[0], [1, 2], [3]] output_shape [1, 4, 32, 32]
      : tensor<1x128x32xf16> into tensor<1x4x32x32xf16>
  %te = tensor.empty() : tensor<1x32x4x32xf16>
  %t = hip.transpose(%ctx) ins(%e : tensor<1x4x32x32xf16>)
      outs(%te : tensor<1x32x4x32xf16>) {perm = [0, 3, 1, 2]} : tensor<1x32x4x32xf16>
  %me = tensor.empty() : tensor<1x4x32x32xf16>
  %m = hip.mul(%ctx) ins(%e, %z : tensor<1x4x32x32xf16>, tensor<1x4x32x32xf16>)
      outs(%me : tensor<1x4x32x32xf16>) -> tensor<1x4x32x32xf16>
  return %t, %m : tensor<1x32x4x32xf16>, tensor<1x4x32x32xf16>
}

// -----

// A crop that the module reads moves into the kernel, which reads the uncropped
// tensor through the slice instead of a strided copy of it.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%[[X:.*]]: tensor<1x8x10x10xf16>, %{{.*}}: tensor<1x8x1x1xf16>) -> tensor<1x8x8x8xf16>
// CHECK-SAME: attributes {rock.elementwise, rock.kernel}
// CHECK: tensor.extract_slice %[[X]][0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
// CHECK: hip.add
// CHECK: hip.leaky_relu
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%{{.*}}, %{{.*}} : tensor<1x8x10x10xf16>, tensor<1x8x1x1xf16>)
// CHECK: return %[[K]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x10x10xf16>,
                      %b: tensor<1x8x1x1xf16>) -> tensor<1x8x8x8xf16> {
  %s = tensor.extract_slice %x[0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
      : tensor<1x8x10x10xf16> to tensor<1x8x8x8xf16>
  %ae = tensor.empty() : tensor<1x8x8x8xf16>
  %a = hip.add(%ctx) ins(%s, %b : tensor<1x8x8x8xf16>, tensor<1x8x1x1xf16>)
      outs(%ae : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8xf16>
  %le = tensor.empty() : tensor<1x8x8x8xf16>
  %l = hip.leaky_relu(%ctx) ins(%a : tensor<1x8x8x8xf16>)
      outs(%le : tensor<1x8x8x8xf16>) {alpha = 1.000000e-01 : f64} : tensor<1x8x8x8xf16>
  return %l : tensor<1x8x8x8xf16>
}

// -----

// A channel slice of an interleaved [N, 2] tensor, with the reshape before it,
// moves into the kernel too.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%[[F:.*]]: tensor<1x8x8x2xf16>, %{{.*}}: tensor<1x64x1xf16>) -> tensor<1x64x1xf16>
// CHECK: tensor.collapse_shape %[[F]]
// CHECK: tensor.extract_slice
// CHECK: hip.add
// CHECK: hip.mul
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%{{.*}}, %{{.*}} : tensor<1x8x8x2xf16>, tensor<1x64x1xf16>)

func.func @main_graph(%ctx: !hip.context, %f: tensor<1x8x8x2xf16>,
                      %g: tensor<1x64x1xf16>) -> tensor<1x64x1xf16> {
  %c = tensor.collapse_shape %f [[0], [1, 2], [3]]
      : tensor<1x8x8x2xf16> into tensor<1x64x2xf16>
  %s = tensor.extract_slice %c[0, 0, 1] [1, 64, 1] [1, 1, 1]
      : tensor<1x64x2xf16> to tensor<1x64x1xf16>
  %ae = tensor.empty() : tensor<1x64x1xf16>
  %a = hip.add(%ctx) ins(%s, %g : tensor<1x64x1xf16>, tensor<1x64x1xf16>)
      outs(%ae : tensor<1x64x1xf16>) -> tensor<1x64x1xf16>
  %me = tensor.empty() : tensor<1x64x1xf16>
  %m = hip.mul(%ctx) ins(%a, %g : tensor<1x64x1xf16>, tensor<1x64x1xf16>)
      outs(%me : tensor<1x64x1xf16>) -> tensor<1x64x1xf16>
  return %m : tensor<1x64x1xf16>
}

// -----

// The graph also returns the crop, so the crop stays for that reader, and the
// kernel reads the uncropped tensor through its own copy of the slice.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%{{.*}}: tensor<1x8x10x10xf16>, %{{.*}}: tensor<1x8x1x1xf16>)
// CHECK: tensor.extract_slice
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[S:.*]] = tensor.extract_slice
// CHECK: %[[K:.*]] = hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%{{.*}}, %{{.*}} : tensor<1x8x10x10xf16>, tensor<1x8x1x1xf16>)
// CHECK: return %[[K]], %[[S]]

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x10x10xf16>,
                      %b: tensor<1x8x1x1xf16>)
    -> (tensor<1x8x8x8xf16>, tensor<1x8x8x8xf16>) {
  %s = tensor.extract_slice %x[0, 0, 1, 1] [1, 8, 8, 8] [1, 1, 1, 1]
      : tensor<1x8x10x10xf16> to tensor<1x8x8x8xf16>
  %ae = tensor.empty() : tensor<1x8x8x8xf16>
  %a = hip.add(%ctx) ins(%s, %b : tensor<1x8x8x8xf16>, tensor<1x8x1x1xf16>)
      outs(%ae : tensor<1x8x8x8xf16>) -> tensor<1x8x8x8xf16>
  %le = tensor.empty() : tensor<1x8x8x8xf16>
  %l = hip.leaky_relu(%ctx) ins(%a : tensor<1x8x8x8xf16>)
      outs(%le : tensor<1x8x8x8xf16>) {alpha = 1.000000e-01 : f64} : tensor<1x8x8x8xf16>
  return %l, %s : tensor<1x8x8x8xf16>, tensor<1x8x8x8xf16>
}

// -----

// rocMLIR has no strided slice, so a slice with stride 2 stays outside, and the
// kernel reads its result.

// CHECK-LABEL: func.func @mlir_{{[a-z0-9_]+}}
// CHECK-SAME: (%{{.*}}: tensor<1x8x4x4xf16>, %{{.*}}: tensor<1x8x1x1xf16>)
// CHECK-NOT: tensor.extract_slice
// CHECK-LABEL: func.func @main_graph
// CHECK: %[[S:.*]] = tensor.extract_slice
// CHECK: hip.rocmlir({{.*}}) @mlir_{{[a-z0-9_]+}} ins(%[[S]],

func.func @main_graph(%ctx: !hip.context, %x: tensor<1x8x8x8xf16>,
                      %b: tensor<1x8x1x1xf16>) -> tensor<1x8x4x4xf16> {
  %s = tensor.extract_slice %x[0, 0, 0, 0] [1, 8, 4, 4] [1, 1, 2, 2]
      : tensor<1x8x8x8xf16> to tensor<1x8x4x4xf16>
  %ae = tensor.empty() : tensor<1x8x4x4xf16>
  %a = hip.add(%ctx) ins(%s, %b : tensor<1x8x4x4xf16>, tensor<1x8x1x1xf16>)
      outs(%ae : tensor<1x8x4x4xf16>) -> tensor<1x8x4x4xf16>
  %le = tensor.empty() : tensor<1x8x4x4xf16>
  %l = hip.leaky_relu(%ctx) ins(%a : tensor<1x8x4x4xf16>)
      outs(%le : tensor<1x8x4x4xf16>) {alpha = 1.000000e-01 : f64} : tensor<1x8x4x4xf16>
  return %l : tensor<1x8x4x4xf16>
}
