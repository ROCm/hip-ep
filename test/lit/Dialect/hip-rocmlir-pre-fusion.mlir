// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip-rocmlir-pre-fusion turns a hip.gather of one constant index on
// float data into a size-1 tensor.extract_slice, plus a tensor.collapse_shape
// for a scalar index, when its one reader will likely run in a rocMLIR
// kernel. The FUSE run checks that the rocMLIR fusion then absorbs the slice.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-rocmlir-pre-fusion %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-fusion %s \
// RUN:   | FileCheck %s --check-prefix=FUSE
// RUN: hip-mlir-opt --split-input-file \
// RUN:   --hip-rocmlir-fusion="standalone-elementwise=false" %s \
// RUN:   | FileCheck %s --check-prefix=NOEW

// Two ops inside one module read the gather. The module takes it as one
// input, but inlined back each op would copy the slice, so the gather stays.
// The NOEW run inlines the module and checks that no slice is left.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

// NOEW-LABEL: func.func @main_graph
// NOEW-NOT: tensor.extract_slice
// NOEW: hip.gather
// NOEW: hip.add
// NOEW: hip.mul

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x64x32xf32>,
                      %b: tensor<64x32xf32>) -> tensor<64x32xf32> {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<64x32xf32>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x64x32xf32>, tensor<i64>)
      outs(%e : tensor<64x32xf32>) {axis = 0 : i64} : tensor<64x32xf32>
  %ae = tensor.empty() : tensor<64x32xf32>
  %a = hip.add(%ctx) ins(%g, %b : tensor<64x32xf32>, tensor<64x32xf32>)
      outs(%ae : tensor<64x32xf32>) -> tensor<64x32xf32>
  %me = tensor.empty() : tensor<64x32xf32>
  %y = hip.mul(%ctx) ins(%g, %a : tensor<64x32xf32>, tensor<64x32xf32>)
      outs(%me : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %y : tensor<64x32xf32>
}

// -----

// An anchor reader: the slice goes into the matmul kernel.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<3x16x32xf16>, %[[W:.*]]: tensor<32x8xf16>)
// CHECK-NEXT: %[[S:.*]] = tensor.extract_slice %[[X]][1, 0, 0] [1, 16, 32] [1, 1, 1] : tensor<3x16x32xf16> to tensor<1x16x32xf16>
// CHECK-NEXT: %[[C:.*]] = tensor.collapse_shape %[[S]] {{\[}}[0, 1], [2]] : tensor<1x16x32xf16> into tensor<16x32xf16>
// CHECK-NEXT: %[[E:.*]] = tensor.empty() : tensor<16x8xf16>
// CHECK-NEXT: %[[Y:.*]] = hip.matmul(%{{.*}}) ins(%[[C]], %[[W]] : tensor<16x32xf16>, tensor<32x8xf16>) outs(%[[E]] : tensor<16x8xf16>) : tensor<16x8xf16>
// CHECK-NEXT: return %[[Y]] : tensor<16x8xf16>

// FUSE-LABEL: func.func @mlir_extract_slice_matmul_0
// FUSE: tensor.extract_slice
// FUSE-LABEL: func.func @main_graph
// FUSE-NOT: hip.gather
// FUSE-NOT: tensor.extract_slice
// FUSE: hip.rocmlir({{.*}}) @mlir_extract_slice_matmul_0

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>) -> tensor<16x8xf16> {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me = tensor.empty() : tensor<16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y : tensor<16x8xf16>
}

// -----

// A module of two compute ops becomes a kernel of its own and takes the
// slice; a one-element 1-D index keeps the unit dim, so only a slice is made.

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%{{.*}}: !hip.context, %[[X:.*]]: tensor<2x64x32xf32>, %[[B:.*]]: tensor<1x64x32xf32>)
// CHECK-NEXT: %[[S:.*]] = tensor.extract_slice %[[X]][1, 0, 0] [1, 64, 32] [1, 1, 1] : tensor<2x64x32xf32> to tensor<1x64x32xf32>
// CHECK-NEXT: hip.rocmlir_elementwise(%{{.*}}) ins(%[[S]], %[[B]] :

// FUSE-LABEL: func.func @mlir_extract_slice_add_neg_0
// FUSE: tensor.extract_slice
// FUSE-LABEL: func.func @main_graph
// FUSE-NOT: hip.gather
// FUSE: hip.rocmlir({{.*}}) @mlir_extract_slice_add_neg_0

func.func @main_graph(%ctx: !hip.context, %x: tensor<2x64x32xf32>,
                      %b: tensor<1x64x32xf32>) -> tensor<1x64x32xf32> {
  %i = hip.constant {value = dense<[1]> : tensor<1xi64>} : tensor<1xi64>
  %e = tensor.empty() : tensor<1x64x32xf32>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<2x64x32xf32>, tensor<1xi64>)
      outs(%e : tensor<1x64x32xf32>) {axis = 0 : i64} : tensor<1x64x32xf32>
  %ae = tensor.empty() : tensor<1x64x32xf32>
  %a = hip.add(%ctx) ins(%g, %b : tensor<1x64x32xf32>, tensor<1x64x32xf32>)
      outs(%ae : tensor<1x64x32xf32>) -> tensor<1x64x32xf32>
  %ne = tensor.empty() : tensor<1x64x32xf32>
  %n = hip.neg(%ctx) ins(%a : tensor<1x64x32xf32>)
      outs(%ne : tensor<1x64x32xf32>) : tensor<1x64x32xf32>
  return %n : tensor<1x64x32xf32>
}

// -----


// A lone pointwise op with no anchor stays a HIP op, so the gather stays.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

// FUSE-LABEL: func.func @main_graph
// FUSE: hip.gather
// FUSE: hip.mul

func.func @main_graph(%ctx: !hip.context, %x: tensor<6x4xf32>)
    -> tensor<6xf32> {
  %i = hip.constant {value = dense<2> : tensor<i64>} : tensor<i64>
  %two = hip.constant {value = dense<2.000000e+00> : tensor<f32>} : tensor<f32>
  %e = tensor.empty() : tensor<6xf32>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<6x4xf32>, tensor<i64>)
      outs(%e : tensor<6xf32>) {axis = 1 : i64} : tensor<6xf32>
  %me = tensor.empty() : tensor<6xf32>
  %y = hip.mul(%ctx) ins(%g, %two : tensor<6xf32>, tensor<f32>)
      outs(%me : tensor<6xf32>) -> tensor<6xf32>
  return %y : tensor<6xf32>
}

// -----

// Two readers: the matmul kernel would not be the slice's only reader.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>)
    -> (tensor<16x8xf16>, tensor<16x32xf16>) {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me = tensor.empty() : tensor<16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y, %g : tensor<16x8xf16>, tensor<16x32xf16>
}

// -----

// Identical gathers stay: CSE would merge their slices into one value with
// two readers.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>)
    -> (tensor<16x8xf16>, tensor<16x8xf16>) {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e0 = tensor.empty() : tensor<16x32xf16>
  %g0 = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e0 : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %e1 = tensor.empty() : tensor<16x32xf16>
  %g1 = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e1 : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me0 = tensor.empty() : tensor<16x8xf16>
  %y0 = hip.matmul(%ctx) ins(%g0, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me0 : tensor<16x8xf16>) : tensor<16x8xf16>
  %me1 = tensor.empty() : tensor<16x8xf16>
  %y1 = hip.matmul(%ctx) ins(%g1, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me1 : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y0, %y1 : tensor<16x8xf16>, tensor<16x8xf16>
}

// -----

// Integer data stays a gather: rocMLIR kernels take floating-point data.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xi32>)
    -> tensor<16x32xi32> {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<16x32xi32>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xi32>, tensor<i64>)
      outs(%e : tensor<16x32xi32>) {axis = 0 : i64} : tensor<16x32xi32>
  return %g : tensor<16x32xi32>
}

// -----

// A runtime index stays a gather.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>, %i: tensor<i64>)
    -> tensor<16x8xf16> {
  %e = tensor.empty() : tensor<16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me = tensor.empty() : tensor<16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y : tensor<16x8xf16>
}

// -----

// An out-of-range index stays a gather, which defines that case.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>) -> tensor<16x8xf16> {
  %i = hip.constant {value = dense<5> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me = tensor.empty() : tensor<16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y : tensor<16x8xf16>
}

// -----

// Several indices stay a gather.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @main_graph(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                      %w: tensor<32x8xf16>) -> tensor<2x16x8xf16> {
  %i = hip.constant {value = dense<[0, 2]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<2x16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<2xi64>)
      outs(%e : tensor<2x16x32xf16>) {axis = 0 : i64} : tensor<2x16x32xf16>
  %me = tensor.empty() : tensor<2x16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<2x16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<2x16x8xf16>) : tensor<2x16x8xf16>
  return %y : tensor<2x16x8xf16>
}

// -----

// Outside main_graph nothing is rewritten.

// CHECK-LABEL: func.func @helper
// CHECK-NOT: tensor.extract_slice
// CHECK: hip.gather

func.func @helper(%ctx: !hip.context, %x: tensor<3x16x32xf16>,
                  %w: tensor<32x8xf16>) -> tensor<16x8xf16> {
  %i = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %e = tensor.empty() : tensor<16x32xf16>
  %g = hip.gather(%ctx) ins(%x, %i : tensor<3x16x32xf16>, tensor<i64>)
      outs(%e : tensor<16x32xf16>) {axis = 0 : i64} : tensor<16x32xf16>
  %me = tensor.empty() : tensor<16x8xf16>
  %y = hip.matmul(%ctx) ins(%g, %w : tensor<16x32xf16>, tensor<32x8xf16>)
      outs(%me : tensor<16x8xf16>) : tensor<16x8xf16>
  return %y : tensor<16x8xf16>
}
