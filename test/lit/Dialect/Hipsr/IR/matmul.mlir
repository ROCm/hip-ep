// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics

// MatMul requires A to provide a contraction dimension.
func.func @matmul_rank0_a(
    %ctx: !hipsr.context, %a: tensor<f16, #hipsr.mem<device>>,
    %b: tensor<4096x1024xf16, #hipsr.mem<device>>,
    %init: tensor<1024xf16, #hipsr.mem<device>>) -> tensor<1024xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %0 = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<f16, #hipsr.mem<device>>, tensor<4096x1024xf16, #hipsr.mem<device>>)
      outs(%init : tensor<1024xf16, #hipsr.mem<device>>) : tensor<1024xf16, #hipsr.mem<device>>
  return %0 : tensor<1024xf16, #hipsr.mem<device>>
}

// -----

// MatMul requires B to provide a contraction dimension.
func.func @matmul_rank0_b(
    %ctx: !hipsr.context, %a: tensor<64x4096xf16, #hipsr.mem<device>>,
    %b: tensor<f16, #hipsr.mem<device>>, %init: tensor<64xf16, #hipsr.mem<device>>) -> tensor<64xf16, #hipsr.mem<device>> {
  // expected-error@+1 {{operand B must be at least 1-D}}
  %0 = hipsr.matmul(%ctx)
      ins(%a, %b : tensor<64x4096xf16, #hipsr.mem<device>>, tensor<f16, #hipsr.mem<device>>)
      outs(%init : tensor<64xf16, #hipsr.mem<device>>) : tensor<64xf16, #hipsr.mem<device>>
  return %0 : tensor<64xf16, #hipsr.mem<device>>
}
