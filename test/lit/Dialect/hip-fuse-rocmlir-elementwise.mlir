// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The second hip-mlir-opt re-parses and re-verifies each module.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir-elementwise %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-inline-rocmlir-elementwise %s \
// RUN:   | FileCheck %s --check-prefix=ROUND

// R1: a pointwise chain is one module, and the anchor stays outside. An
// inline scalar constant stays outside as an input, so its
// serialization_order is never cloned.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[ZERO:.*]] = hip.constant
// CHECK: %[[C:.*]] = hip.conv
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%[[C]], %{{.*}}, %[[ZERO]] : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>, tensor<f16>) -> tensor<1x16x4x4xf16> {
// CHECK: ^bb0(%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x16x4x4xf16>, %[[R:.*]]: tensor<1x16x4x4xf16>, %[[Z:.*]]: tensor<f16>):
// CHECK: %[[AE:.*]] = tensor.empty() : tensor<1x16x4x4xf16>
// CHECK: %[[A:.*]] = hip.add(%[[CTX]]) ins(%[[X]], %[[R]] : {{.*}}) outs(%[[AE]] :
// CHECK: %[[MX:.*]] = hip.max(%[[CTX]]) ins(%[[A]], %[[Z]] :
// CHECK: hip.yield %[[MX]]
// CHECK: return %[[M]]

// The inline pass puts the ops back.
// ROUND-LABEL: func.func @main_graph
// ROUND-NOT: hip.rocmlir_elementwise
// ROUND: %[[C:.*]] = hip.conv
// ROUND: %[[A:.*]] = hip.add(%{{.*}}) ins(%[[C]],
// ROUND: %[[MX:.*]] = hip.max(%{{.*}}) ins(%[[A]],
// ROUND: return %[[MX]]
func.func @main_graph(%ctx: !hip.context, %in: tensor<1x8x4x4xf16>,
                      %w: tensor<16x8x3x3xf16>, %b: tensor<16xf16>,
                      %r: tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16> {
  %zero = hip.constant {value = dense<0.000000e+00> : tensor<f16>} : tensor<f16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%in, %w, %b : tensor<1x8x4x4xf16>,
                                        tensor<16x8x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%c, %r : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.max(%ctx) ins(%a, %zero : tensor<1x16x4x4xf16>, tensor<f16>)
      outs(%me : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// R2: a broadcast stays outside, as a module input.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[SB:.*]] = hip.expand
// CHECK: hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %[[SB]] :
// CHECK: hip.mul
// CHECK-NOT: hip.expand
// CHECK: hip.yield
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x4x4xf16>,
                      %s: tensor<1x16x1x1xf16>) -> tensor<1x16x4x4xf16> {
  %shape = hip.constant {value = dense<[1, 16, 4, 4]> : tensor<4xi64>} : tensor<4xi64>
  %se = tensor.empty() : tensor<1x16x4x4xf16>
  %sb = hip.expand(%ctx) ins(%s, %shape : tensor<1x16x1x1xf16>, tensor<4xi64>)
      outs(%se : tensor<1x16x4x4xf16>) : tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%x, %sb : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// R3: an op that broadcasts a module value starts a new module, because
// rocMLIR cannot invert the broadcast in an anchor epilogue.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1x1x16xf16>, tensor<1x1x16xf16>) -> tensor<1x1x16xf16>
// CHECK: hip.mul
// CHECK-NOT: hip.add
// CHECK: hip.yield
// CHECK: %[[A:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%[[M]], %{{.*}} : tensor<1x1x16xf16>, tensor<1x2x16xf16>) -> tensor<1x2x16xf16>
// CHECK: hip.add
// CHECK: return %[[A]]
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x1x16xf16>,
                      %y: tensor<1x1x16xf16>, %z: tensor<1x2x16xf16>)
    -> tensor<1x2x16xf16> {
  %me = tensor.empty() : tensor<1x1x16xf16>
  %m = hip.mul(%ctx) ins(%x, %y : tensor<1x1x16xf16>, tensor<1x1x16xf16>)
      outs(%me : tensor<1x1x16xf16>) -> tensor<1x1x16xf16>
  %ae = tensor.empty() : tensor<1x2x16xf16>
  %a = hip.add(%ctx) ins(%m, %z : tensor<1x1x16xf16>, tensor<1x2x16xf16>)
      outs(%ae : tensor<1x2x16xf16>) -> tensor<1x2x16xf16>
  return %a : tensor<1x2x16xf16>
}

// -----

// R4: a value with a reader outside is one more module result. A reshape
// read outside has no init to write, so it stays outside.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[G:.*]]:2 = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x16xf16>, tensor<4x16xf16>) -> tensor<4x16xf16>, tensor<4x16xf16>
// CHECK: %[[A:.*]] = hip.add
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[A]], %[[A]] :
// CHECK: hip.yield %[[A]], %[[M]]
// CHECK: %[[V:.*]] = tensor.collapse_shape %[[G]]#1
// CHECK: return %[[G]]#0, %[[G]]#1, %[[V]]
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<4x16xf16>,
                      %y: tensor<4x16xf16>)
    -> (tensor<4x16xf16>, tensor<4x16xf16>, tensor<64xf16>) {
  %ae = tensor.empty() : tensor<4x16xf16>
  %a = hip.add(%ctx) ins(%x, %y : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%ae : tensor<4x16xf16>) -> tensor<4x16xf16>
  %me = tensor.empty() : tensor<4x16xf16>
  %m = hip.mul(%ctx) ins(%a, %a : tensor<4x16xf16>, tensor<4x16xf16>)
      outs(%me : tensor<4x16xf16>) -> tensor<4x16xf16>
  %v = tensor.collapse_shape %m [[0, 1]] : tensor<4x16xf16> into tensor<64xf16>
  return %a, %m, %v : tensor<4x16xf16>, tensor<4x16xf16>, tensor<64xf16>
}

// -----

// R5: a dequantize starts a module and a quantize ends one, so the module
// reads and writes 8-bit values.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[Q:.*]] = hip.rocmlir_elementwise({{.*}}) -> tensor<1x16x4x4xi8>
// CHECK: hip.dequantize_linear
// CHECK: hip.add
// CHECK: %[[QV:.*]] = hip.quantize_linear
// CHECK: hip.yield %[[QV]]
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%[[Q]],
// CHECK: hip.dequantize_linear
// CHECK: hip.mul
// CHECK: return %[[M]]

// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %xq: tensor<1x16x4x4xi8>,
                      %r: tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32> {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %de = tensor.empty() : tensor<1x16x4x4xf32>
  %d = hip.dequantize_linear(%ctx) ins(%xq, %s : tensor<1x16x4x4xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%de : tensor<1x16x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x16x4x4xf32>
  %ae = tensor.empty() : tensor<1x16x4x4xf32>
  %a = hip.add(%ctx) ins(%d, %r : tensor<1x16x4x4xf32>, tensor<1x16x4x4xf32>)
      outs(%ae : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  %qe = tensor.empty() : tensor<1x16x4x4xi8>
  %q = hip.quantize_linear(%ctx) ins(%a, %s : tensor<1x16x4x4xf32>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%qe : tensor<1x16x4x4xi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<1x16x4x4xi8>
  %d2e = tensor.empty() : tensor<1x16x4x4xf32>
  %d2 = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<1x16x4x4xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%d2e : tensor<1x16x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x16x4x4xf32>
  %me = tensor.empty() : tensor<1x16x4x4xf32>
  %m = hip.mul(%ctx) ins(%d2, %r : tensor<1x16x4x4xf32>, tensor<1x16x4x4xf32>)
      outs(%me : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  return %m : tensor<1x16x4x4xf32>
}

// -----

// R6: an i64 input keeps its op outside. An i64 result read outside cannot
// be stored by the kernel, so the cast that makes it stays outside.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4x8xi64>, tensor<4x8xi64>)
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}} : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK: hip.mul
// CHECK-NOT: hip.cast
// CHECK: hip.yield
// CHECK: %[[I:.*]] = hip.cast({{.*}}) ins(%[[M]] :
// CHECK: return %[[A]], %[[I]]
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<4x8xi64>,
                      %y: tensor<4x8xf32>) -> (tensor<4x8xi64>, tensor<4x8xi64>) {
  %ae = tensor.empty() : tensor<4x8xi64>
  %a = hip.add(%ctx) ins(%x, %x : tensor<4x8xi64>, tensor<4x8xi64>)
      outs(%ae : tensor<4x8xi64>) -> tensor<4x8xi64>
  %me = tensor.empty() : tensor<4x8xf32>
  %m = hip.mul(%ctx) ins(%y, %y : tensor<4x8xf32>, tensor<4x8xf32>)
      outs(%me : tensor<4x8xf32>) -> tensor<4x8xf32>
  %ie = tensor.empty() : tensor<4x8xi64>
  %i = hip.cast(%ctx) ins(%m : tensor<4x8xf32>)
      outs(%ie : tensor<4x8xi64>) {to = 7 : i64} : tensor<4x8xi64>
  return %a, %i : tensor<4x8xi64>, tensor<4x8xi64>
}

// -----

// R7: views alone make no module.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir_elementwise
// CHECK: hip.transpose
// CHECK: tensor.collapse_shape
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<16x8xf16>)
    -> tensor<128xf16> {
  %te = tensor.empty() : tensor<8x16xf16>
  %t = hip.transpose(%ctx) ins(%x : tensor<16x8xf16>)
      outs(%te : tensor<8x16xf16>) {perm = [1, 0]} : tensor<8x16xf16>
  %v = tensor.collapse_shape %t [[0, 1]] : tensor<8x16xf16> into tensor<128xf16>
  return %v : tensor<128xf16>
}

// -----

// R8: an op whose other input depends on the module through an outside op
// starts a new module, so no module reads its own result.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[A:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}} : tensor<1x16x4x4xf16>)
// CHECK: hip.add
// CHECK-NOT: hip.mul
// CHECK: hip.yield
// CHECK: %[[C:.*]] = hip.conv({{.*}}) ins(%[[A]],
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%[[A]], %[[C]] :
// CHECK: hip.mul
// CHECK: return %[[M]]
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x4x4xf16>,
                      %w: tensor<16x16x3x3xf16>, %b: tensor<16xf16>)
    -> tensor<1x16x4x4xf16> {
  %ae = tensor.empty() : tensor<1x16x4x4xf16>
  %a = hip.add(%ctx) ins(%x, %x : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%ae : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  %e = tensor.empty() : tensor<1x16x4x4xf16>
  %c = hip.conv(%ctx) ins(%a, %w, %b : tensor<1x16x4x4xf16>,
                                       tensor<16x16x3x3xf16>, tensor<16xf16>)
      outs(%e : tensor<1x16x4x4xf16>)
      {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3],
       pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x16x4x4xf16>
  %me = tensor.empty() : tensor<1x16x4x4xf16>
  %m = hip.mul(%ctx) ins(%a, %c : tensor<1x16x4x4xf16>, tensor<1x16x4x4xf16>)
      outs(%me : tensor<1x16x4x4xf16>) -> tensor<1x16x4x4xf16>
  return %m : tensor<1x16x4x4xf16>
}

// -----

// A transpose and a static reshape between pointwise ops stay in the module.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[G:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %{{.*}} : tensor<16x8xf16>, tensor<128xf16>) -> tensor<128xf16>
// CHECK: %[[N:.*]] = hip.neg
// CHECK: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[N]] :
// CHECK: %[[V:.*]] = tensor.collapse_shape %[[T]]
// CHECK: %[[A:.*]] = hip.add({{.*}}) ins(%[[V]],
// CHECK: hip.yield %[[A]]
// CHECK: return %[[G]]

// ROUND-LABEL: func.func @main_graph
// ROUND-NOT: hip.rocmlir_elementwise
// ROUND: %[[N:.*]] = hip.neg
// ROUND: %[[T:.*]] = hip.transpose({{.*}}) ins(%[[N]] :
// ROUND: %[[V:.*]] = tensor.collapse_shape %[[T]]
// ROUND: %[[A:.*]] = hip.add({{.*}}) ins(%[[V]],
// ROUND: return %[[A]]
func.func @main_graph(%ctx: !hip.context, %x: tensor<16x8xf16>,
                      %y: tensor<128xf16>) -> tensor<128xf16> {
  %ne = tensor.empty() : tensor<16x8xf16>
  %n = hip.neg(%ctx) ins(%x : tensor<16x8xf16>)
      outs(%ne : tensor<16x8xf16>) : tensor<16x8xf16>
  %te = tensor.empty() : tensor<8x16xf16>
  %t = hip.transpose(%ctx) ins(%n : tensor<16x8xf16>)
      outs(%te : tensor<8x16xf16>) {perm = [1, 0]} : tensor<8x16xf16>
  %v = tensor.collapse_shape %t [[0, 1]] : tensor<8x16xf16> into tensor<128xf16>
  %ae = tensor.empty() : tensor<128xf16>
  %a = hip.add(%ctx) ins(%v, %y : tensor<128xf16>, tensor<128xf16>)
      outs(%ae : tensor<128xf16>) -> tensor<128xf16>
  return %a : tensor<128xf16>
}

// -----

// R5: a dequantize that only the module reads joins it from above, also when
// it comes after the module's first op, so the module loads the 8-bit input.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.dequantize_linear
// CHECK: %[[M:.*]] = hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x16x4x4xf32>, tensor<1x16x4x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x16x4x4xf32>
// CHECK: hip.neg
// CHECK: hip.dequantize_linear
// CHECK: hip.add
// CHECK: return %[[M]]
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x4x4xf32>,
                      %rq: tensor<1x16x4x4xi8>) -> tensor<1x16x4x4xf32> {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %ne = tensor.empty() : tensor<1x16x4x4xf32>
  %n = hip.neg(%ctx) ins(%x : tensor<1x16x4x4xf32>)
      outs(%ne : tensor<1x16x4x4xf32>) : tensor<1x16x4x4xf32>
  %de = tensor.empty() : tensor<1x16x4x4xf32>
  %d = hip.dequantize_linear(%ctx) ins(%rq, %s : tensor<1x16x4x4xi8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%de : tensor<1x16x4x4xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<1x16x4x4xf32>
  %ae = tensor.empty() : tensor<1x16x4x4xf32>
  %a = hip.add(%ctx) ins(%n, %d : tensor<1x16x4x4xf32>, tensor<1x16x4x4xf32>)
      outs(%ae : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  return %a : tensor<1x16x4x4xf32>
}

// -----

// A rank-0 value gets no role: the dequantize of a scalar constant stays
// outside, as a module input.

// CHECK-LABEL: func.func @main_graph
// CHECK: %[[D:.*]] = hip.dequantize_linear
// CHECK: hip.rocmlir_elementwise(%{{.*}}) ins(%{{.*}}, %[[D]] : tensor<1x16x4x4xf32>, tensor<f32>) -> tensor<1x16x4x4xf32>
// CHECK-NOT: hip.dequantize_linear
// CHECK: hip.mul
// CHECK: hip.yield
// ROUND-LABEL: func.func @main_graph
func.func @main_graph(%ctx: !hip.context, %x: tensor<1x16x4x4xf32>)
    -> tensor<1x16x4x4xf32> {
  %s = hip.constant {value = dense<1.250000e-02> : tensor<f32>} : tensor<f32>
  %zp = hip.constant {value = dense<0> : tensor<i8>} : tensor<i8>
  %q = hip.constant {value = dense<8> : tensor<i8>} : tensor<i8>
  %de = tensor.empty() : tensor<f32>
  %d = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<i8>, tensor<f32>)
      zero_point(%zp : tensor<i8>) outs(%de : tensor<f32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<f32>
  %me = tensor.empty() : tensor<1x16x4x4xf32>
  %m = hip.mul(%ctx) ins(%x, %d : tensor<1x16x4x4xf32>, tensor<f32>)
      outs(%me : tensor<1x16x4x4xf32>) -> tensor<1x16x4x4xf32>
  return %m : tensor<1x16x4x4xf32>
}
