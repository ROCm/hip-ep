// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Gemm-pointwise-gemm: two outlined matmul kernels merge into one when the
// first kernel's only result feeds the second gemm's A operand and MIGraphX's
// shape gates pass. fuse-geg=false keeps them apart.

// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-fuse-rocmlir="fuse-geg=false" %s \
// RUN:   | hip-mlir-opt --split-input-file | FileCheck %s --check-prefix=NOGEG

// A column-vector intermediate (n == 1) passes the gate: one kernel runs
// matmul -> sigmoid -> matmul, and the dispatch reads x and both weights.

// CHECK-LABEL: func.func @rocMlir
// CHECK-SAME: (%[[X:.*]]: tensor<512x64xf16>, %[[W0:.*]]: tensor<64x1xf16>, %[[W1:.*]]: tensor<1x32xf16>) -> tensor<512x32xf16>
// CHECK: %[[M0:.*]] = hip.matmul({{.*}}) ins(%[[X]], %[[W0]]
// CHECK: %[[S:.*]] = hip.sigmoid({{.*}}) ins(%[[M0]]
// CHECK: %[[M1:.*]] = hip.matmul({{.*}}) ins(%[[S]], %[[W1]]
// CHECK: return %[[M1]]

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<512x64xf16>, tensor<64x1xf16>, tensor<1x32xf16>)
// CHECK-NOT: hip.rocmlir

// NOGEG-LABEL: func.func @main_graph
// NOGEG: hip.rocmlir
// NOGEG: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf16>,
                      %w0: tensor<64x1xf16>, %w1: tensor<1x32xf16>)
    -> tensor<512x32xf16> {
  %e0 = tensor.empty() : tensor<512x1xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf16>, tensor<64x1xf16>)
      outs(%e0 : tensor<512x1xf16>) : tensor<512x1xf16>
  %se = tensor.empty() : tensor<512x1xf16>
  %s = hip.sigmoid(%ctx) ins(%m0 : tensor<512x1xf16>)
      outs(%se : tensor<512x1xf16>) : tensor<512x1xf16>
  %e1 = tensor.empty() : tensor<512x32xf16>
  %m1 = hip.matmul(%ctx) ins(%s, %w1 : tensor<512x1xf16>, tensor<1x32xf16>)
      outs(%e1 : tensor<512x32xf16>) : tensor<512x32xf16>
  return %m1 : tensor<512x32xf16>
}

// -----

// A wide intermediate (n = 64) fails the gate: the m x n tile is cheap to
// round-trip, so the kernels stay apart.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf16>,
                      %w0: tensor<64x64xf16>, %w1: tensor<64x32xf16>)
    -> tensor<512x32xf16> {
  %e0 = tensor.empty() : tensor<512x64xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf16>, tensor<64x64xf16>)
      outs(%e0 : tensor<512x64xf16>) : tensor<512x64xf16>
  %e1 = tensor.empty() : tensor<512x32xf16>
  %m1 = hip.matmul(%ctx) ins(%m0, %w1 : tensor<512x64xf16>, tensor<64x32xf16>)
      outs(%e1 : tensor<512x32xf16>) : tensor<512x32xf16>
  return %m1 : tensor<512x32xf16>
}

// -----

// WMMA has no fp32 path on gfx11, so an fp32 pair stays apart.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf32>,
                      %w0: tensor<64x1xf32>, %w1: tensor<1x32xf32>)
    -> tensor<512x32xf32> {
  %e0 = tensor.empty() : tensor<512x1xf32>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf32>, tensor<64x1xf32>)
      outs(%e0 : tensor<512x1xf32>) : tensor<512x1xf32>
  %e1 = tensor.empty() : tensor<512x32xf32>
  %m1 = hip.matmul(%ctx) ins(%m0, %w1 : tensor<512x1xf32>, tensor<1x32xf32>)
      outs(%e1 : tensor<512x32xf32>) : tensor<512x32xf32>
  return %m1 : tensor<512x32xf32>
}

// -----

// The first kernel's result has a second reader, so it must still be written
// out and the kernels stay apart.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf16>,
                      %w0: tensor<64x1xf16>, %w1: tensor<1x32xf16>)
    -> (tensor<512x32xf16>, tensor<512x1xf16>) {
  %e0 = tensor.empty() : tensor<512x1xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf16>, tensor<64x1xf16>)
      outs(%e0 : tensor<512x1xf16>) : tensor<512x1xf16>
  %e1 = tensor.empty() : tensor<512x32xf16>
  %m1 = hip.matmul(%ctx) ins(%m0, %w1 : tensor<512x1xf16>, tensor<1x32xf16>)
      outs(%e1 : tensor<512x32xf16>) : tensor<512x32xf16>
  return %m1, %m0 : tensor<512x32xf16>, tensor<512x1xf16>
}

// -----

// A reshape that only regroups the rows (batch and m) may sit between the
// gemms, as in a 2-D gemm whose result feeds a batched matmul.

// CHECK-LABEL: func.func @rocMlir
// CHECK: hip.matmul
// CHECK: tensor.expand_shape
// CHECK: hip.matmul
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf16>,
                      %w0: tensor<64x1xf16>, %w1: tensor<1x32xf16>)
    -> tensor<2x256x32xf16> {
  %e0 = tensor.empty() : tensor<512x1xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf16>, tensor<64x1xf16>)
      outs(%e0 : tensor<512x1xf16>) : tensor<512x1xf16>
  %r = tensor.expand_shape %m0 [[0, 1], [2]] output_shape [2, 256, 1]
      : tensor<512x1xf16> into tensor<2x256x1xf16>
  %e1 = tensor.empty() : tensor<2x256x32xf16>
  %m1 = hip.matmul(%ctx) ins(%r, %w1 : tensor<2x256x1xf16>, tensor<1x32xf16>)
      outs(%e1 : tensor<2x256x32xf16>) : tensor<2x256x32xf16>
  return %m1 : tensor<2x256x32xf16>
}

// -----

// A transpose between the gemms would turn the first gemm's n into the second
// gemm's rows, which rocMLIR's gemm-elementwise-gemm cannot express.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<256x64xf16>,
                      %w0: tensor<64x1xf16>, %w1: tensor<256x32xf16>)
    -> tensor<1x32xf16> {
  %e0 = tensor.empty() : tensor<256x1xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<256x64xf16>, tensor<64x1xf16>)
      outs(%e0 : tensor<256x1xf16>) : tensor<256x1xf16>
  %te = tensor.empty() : tensor<1x256xf16>
  %t = hip.transpose(%ctx) ins(%m0 : tensor<256x1xf16>)
      outs(%te : tensor<1x256xf16>) {perm = [1, 0]} : tensor<1x256xf16>
  %e1 = tensor.empty() : tensor<1x32xf16>
  %m1 = hip.matmul(%ctx) ins(%t, %w1 : tensor<1x256xf16>, tensor<256x32xf16>)
      outs(%e1 : tensor<1x32xf16>) : tensor<1x32xf16>
  return %m1 : tensor<1x32xf16>
}
