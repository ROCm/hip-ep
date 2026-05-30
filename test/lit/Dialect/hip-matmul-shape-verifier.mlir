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
  // expected-error @below {{dim 0 of result #0 mismatch: expected 2}}
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
  // expected-error @below {{dim 1 of result #0 mismatch: expected 8}}
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
  // expected-error @below {{matmul batch dim mismatch at position 0: A=2 B=3}}
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
  // expected-error @below {{rank mismatch on result #0: expected rank 3}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x4x8xf16, 1>, memref<8x16xf16, 1>)
    outs(%c : memref<4x16xf16, 1>)
  return
}

// -----

// B's rank exceeds A's: codegen derives result rank from A and would
// miscompute. Rejected.
func.func @matmul_b_rank_exceeds_a(%ctx: !hip.context,
                                   %a: memref<4x8xf16, 1>,
                                   %b: memref<2x8x16xf16, 1>,
                                   %c: memref<2x4x16xf16, 1>) {
  // expected-error @below {{matmul B's rank (3) exceeds A's rank (2)}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<4x8xf16, 1>, memref<2x8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
  return
}

// -----

// Mixed-rank operands where B isn't exactly 2D: codegen handles only the
// `B == rank-2` broadcast case. Rejected.
func.func @matmul_mixed_rank_b_not_2d(%ctx: !hip.context,
                                      %a: memref<2x3x4x8xf16, 1>,
                                      %b: memref<3x8x16xf16, 1>,
                                      %c: memref<2x3x4x16xf16, 1>) {
  // expected-error @below {{matmul mixed-rank operands require B to be rank-2 (got A rank 4, B rank 3)}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<2x3x4x8xf16, 1>, memref<3x8x16xf16, 1>)
    outs(%c : memref<2x3x4x16xf16, 1>)
  return
}

// -----

// Per-dim batch broadcasting (1 vs >1): codegen reads A's batch verbatim,
// so result batch would be 1 instead of 2. Rejected.
func.func @matmul_per_dim_broadcast_unsupported(%ctx: !hip.context,
                                                %a: memref<1x4x8xf16, 1>,
                                                %b: memref<2x8x16xf16, 1>,
                                                %c: memref<2x4x16xf16, 1>) {
  // expected-error @below {{matmul batch dim mismatch at position 0: A=1 B=2; per-dim batch broadcasting (1 vs >1) is not supported by codegen}}
  hip.matmul(%ctx)
    ins(%a, %b : memref<1x4x8xf16, 1>, memref<2x8x16xf16, 1>)
    outs(%c : memref<2x4x16xf16, 1>)
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
