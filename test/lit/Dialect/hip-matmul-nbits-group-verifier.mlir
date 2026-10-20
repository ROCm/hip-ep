// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @silu_mul
// CHECK:         hip.matmul_nbits_group
// CHECK-SAME:      up_tmp(
func.func @silu_mul(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                    %aq: memref<1x1x64xi8, 1>, %as: memref<1x1x2xf32, 1>,
                    %b: memref<32x2x16xui8, 1>, %s: memref<32x2xf16, 1>,
                    %y: memref<1x1x32xf16, 1>, %t: memref<1x1x32xf16, 1>) {
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      weights(%b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      outs(%y : memref<1x1x32xf16, 1>) up_tmp(%t : memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 32], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, epilogue = 1 : i64}
  return
}

// -----

func.func @four_segments(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                         %aq: memref<1x1x64xi8, 1>, %as: memref<1x1x2xf32, 1>,
                         %b: memref<32x2x16xui8, 1>, %s: memref<32x2xf16, 1>,
                         %y: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{expected 1 to 3 weights, got 4}}
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      weights(%b, %b, %b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>,
              memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s, %s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>,
             memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      outs(%y, %y, %y, %y : memref<1x1x32xf16, 1>, memref<1x1x32xf16, 1>,
           memref<1x1x32xf16, 1>, memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 32, 32, 32], bits = 4 : i64,
       block_size = 32 : i64, accuracy_level = 4 : i64}
  return
}

// -----

func.func @partial_zero_points(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                               %aq: memref<1x1x64xi8, 1>,
                               %as: memref<1x1x2xf32, 1>,
                               %b: memref<32x2x16xui8, 1>,
                               %s: memref<32x2xf16, 1>,
                               %z: memref<32x1xui8, 1>,
                               %y: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{expected no zero_points or one per weight}}
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      weights(%b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      zero_points(%z : memref<32x1xui8, 1>)
      outs(%y, %y : memref<1x1x32xf16, 1>, memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 32], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64}
  return
}

// -----

func.func @silu_without_up_tmp(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                               %aq: memref<1x1x64xi8, 1>,
                               %as: memref<1x1x2xf32, 1>,
                               %b: memref<32x2x16xui8, 1>,
                               %s: memref<32x2xf16, 1>,
                               %y: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{silu_mul epilogue expects 2 weights, 1 output and up_tmp}}
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      weights(%b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      outs(%y : memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 32], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, epilogue = 1 : i64}
  return
}

// -----

func.func @silu_unequal_n(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                          %aq: memref<1x1x64xi8, 1>, %as: memref<1x1x2xf32, 1>,
                          %b: memref<32x2x16xui8, 1>, %s: memref<32x2xf16, 1>,
                          %y: memref<1x1x32xf16, 1>,
                          %t: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{silu_mul epilogue expects equal N}}
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
      weights(%b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      outs(%y : memref<1x1x32xf16, 1>) up_tmp(%t : memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 16], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, epilogue = 1 : i64}
  return
}

// -----

func.func @bad_scale_block(%ctx: !hip.context, %a: memref<1x1x64xf16, 1>,
                           %aq: memref<1x1x64xi8, 1>, %as: memref<1x1x1xf32, 1>,
                           %b: memref<32x2x16xui8, 1>, %s: memref<32x2xf16, 1>,
                           %y: memref<1x1x32xf16, 1>) {
  // expected-error @+1 {{a_scale trailing dim must be K / block size}}
  hip.matmul_nbits_group(%ctx) ins(%a : memref<1x1x64xf16, 1>)
      quantized(%aq, %as : memref<1x1x64xi8, 1>, memref<1x1x1xf32, 1>)
      weights(%b, %b : memref<32x2x16xui8, 1>, memref<32x2x16xui8, 1>)
      scales(%s, %s : memref<32x2xf16, 1>, memref<32x2xf16, 1>)
      outs(%y, %y : memref<1x1x32xf16, 1>, memref<1x1x32xf16, 1>)
      {K = 64 : i64, N = [32, 32], bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64}
  return
}
