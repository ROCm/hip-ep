// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_2d_static
// CHECK:         hip.matmul
func.func @matmul_2d_static(%ctx: !hip.context,
                            %a: memref<128x4096xf32, 1>,
                            %b: memref<4096x1024xf32, 1>,
                            %c: memref<128x1024xf32, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<128x4096xf32, 1>, memref<4096x1024xf32, 1>)
    outs(%c : memref<128x1024xf32, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @matmul_3d_broadcast
// CHECK:         hip.matmul
func.func @matmul_3d_broadcast(%ctx: !hip.context,
                               %a: memref<2x4x8xf16, 1>,
                               %b: memref<8x16xf16, 1>,
                               %c: memref<2x4x16xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @matmul_dynamic_batch
// CHECK:         hip.matmul
func.func @matmul_dynamic_batch(%ctx: !hip.context,
                                %a: memref<?x4x8xf16, 1>,
                                %b: memref<8x16xf16, 1>,
                                %c: memref<?x4x16xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<?x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<?x4x16xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @matmul_dynamic_k
// CHECK:         hip.matmul
func.func @matmul_dynamic_k(%ctx: !hip.context,
                            %a: memref<2x?xf16, 1>,
                            %b: memref<?x8xf16, 1>,
                            %c: memref<2x8xf16, 1>) {
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x?xf16, 1>, memref<?x8xf16, 1>)
    outs(%c : memref<2x8xf16, 1>)
  return
}

// -----

func.func @matmul_k_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<8x16xf16, 1>,
                             %c: memref<2x16xf16, 1>) {
  // expected-error @below {{matmul contraction dim mismatch: A.shape[-1]=4 vs B.shape[-2]=8}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<2x16xf16, 1>)
  return
}

// -----

func.func @matmul_m_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<4x8xf16, 1>,
                             %c: memref<3x8xf16, 1>) {
  // expected-error @below {{dim 0 of result mismatch: expected 2}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>)
    outs(%c : memref<3x8xf16, 1>)
  return
}

// -----

func.func @matmul_n_mismatch(%ctx: !hip.context,
                             %a: memref<2x4xf16, 1>,
                             %b: memref<4x8xf16, 1>,
                             %c: memref<2x9xf16, 1>) {
  // expected-error @below {{dim 1 of result mismatch: expected 8}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4xf16, 1>, memref<4x8xf16, 1>)
    outs(%c : memref<2x9xf16, 1>)
  return
}

// -----

func.func @matmul_batch_broadcast_fail(%ctx: !hip.context,
                                       %a: memref<2x4x8xf16, 1>,
                                       %b: memref<3x8x16xf16, 1>,
                                       %c: memref<2x4x16xf16, 1>) {
  // expected-error @below {{matmul batch broadcast failure}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<3x8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
  return
}

// -----

func.func @matmul_rank_mismatch(%ctx: !hip.context,
                                %a: memref<2x4x8xf16, 1>,
                                %b: memref<8x16xf16, 1>,
                                %c: memref<4x16xf16, 1>) {
  // expected-error @below {{rank mismatch on result: expected rank 3}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<4x16xf16, 1>)
  return
}

// -----

// CHECK-LABEL: func.func @matmul_tensor_mode_static
// CHECK:         hip.matmul
func.func @matmul_tensor_mode_static(%ctx: !hip.context,
                                     %a: tensor<2x4xf16>,
                                     %b: tensor<4x8xf16>,
                                     %c: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.matmul(%ctx)
    ins(%a, %b : tensor<2x4xf16>, tensor<4x8xf16>)
    outs(%c : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
