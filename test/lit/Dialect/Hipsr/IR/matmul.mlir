// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// MatMul requires A to provide a contraction dimension.
func.func @matmul_rank0_a(
    %ctx: !hipsr.context, %a: tensor<f16>,
    %b: tensor<4096x1024xf16>,
    %init: tensor<1024xf16>) -> tensor<1024xf16> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %0 = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<f16>, tensor<4096x1024xf16>)
      outs(%init : tensor<1024xf16>) : tensor<1024xf16>
  return %0 : tensor<1024xf16>
}

// -----

// MatMul requires B to provide a contraction dimension.
func.func @matmul_rank0_b(
    %ctx: !hipsr.context, %a: tensor<64x4096xf16>,
    %b: tensor<f16>, %init: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error@+1 {{operand B must be at least 1-D}}
  %0 = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<64x4096xf16>, tensor<f16>)
      outs(%init : tensor<64xf16>) : tensor<64xf16>
  return %0 : tensor<64xf16>
}
