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
