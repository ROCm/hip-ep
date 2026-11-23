// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="arch=gfx1151 gemm-gemm-accel-types=f16,bf16" %s \
// RUN:   | hip-mlir-opt --split-input-file \
// RUN:   | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | hip-mlir-opt --split-input-file \
// RUN:   | FileCheck %s --check-prefixes=NOGEG,NOARCH





// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
func.func @main_graph
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




// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
func.func @main_graph
// NOGEG: hip.rocmlir
// NOGEG: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %x: tensor<512x64xf16>,
                      %w0: tensor<1x64xf16>, %w1: tensor<1x32xf16>)
    -> tensor<512x32xf16> {
  %e0 = tensor.empty() : tensor<512x1xf16>
  %m0 = hip.matmul(%ctx) ins(%x, %w0 : tensor<512x64xf16>, tensor<1x64xf16>)
      outs(%e0 : tensor<512x1xf16>) {transA = 0 : i64, transB = 1 : i64}
      : tensor<512x1xf16>
  %se = tensor.empty() : tensor<512x1xf16>
  %s = hip.sigmoid(%ctx) ins(%m0 : tensor<512x1xf16>)
      outs(%se : tensor<512x1xf16>) : tensor<512x1xf16>
  %e1 = tensor.empty() : tensor<512x32xf16>
  %m1 = hip.matmul(%ctx) ins(%s, %w1 : tensor<512x1xf16>, tensor<1x32xf16>)
      outs(%e1 : tensor<512x32xf16>) : tensor<512x32xf16>
  return %m1 : tensor<512x32xf16>
}

// -----



// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
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



// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
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



// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
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



// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
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



// Dense GEMMs remain on hip_gemm; the surrounding operations stay in main_graph.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
// CHECK: hip.matmul
// CHECK-NOT: hip.rocmlir
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
