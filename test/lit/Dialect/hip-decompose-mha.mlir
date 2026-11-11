// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-decompose-multi-head-attention %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DECOMPOSE
// RUN: hip-mlir-opt --split-input-file --hip-decompose-multi-head-attention \
// RUN:   --hip-fuse-rocmlir %s | hip-mlir-opt --split-input-file \
// RUN:   | FileCheck %s --check-prefixes=CHECK,FUSE

// f16 prefill, head dim 8. Padded head dim is 16, so the runtime may run this
// as flash attention on wave32. The pass cannot see the device and leaves it.
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.multi_head_attention
// DECOMPOSE-NOT: hip.matmul
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                      %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>)
    -> tensor<1x4x16xf16> {
  %empty = tensor.empty() : tensor<1x4x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>,
      tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%empty : tensor<1x4x16xf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}

// -----

// Decode is not the flash prefill path, so f16 query sequence 1 still splits
// and each matmul is outlined around the softmax. hip-fuse-rocmlir only
// outlines func.func @main_graph, so this case uses that name.
// CHECK-LABEL: func.func @main_graph({{.*}}tensor<1x1x16xf16>, {{.*}}tensor<1x4x16xf16>
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: hip.matmul
// DECOMPOSE: hip.mul
// DECOMPOSE: hip.miopen.softmax
// DECOMPOSE: hip.matmul
// FUSE: hip.rocmlir
// FUSE: hip.miopen.softmax
// FUSE: hip.rocmlir
// FUSE-NOT: hip.multi_head_attention
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                  %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>)
    -> tensor<1x1x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x1x16xf16>,
      tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%empty : tensor<1x1x16xf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x1x16xf16>
  return %r : tensor<1x1x16xf16>
}

// -----

// Cross attention with query sequence 1. A longer query would be f16 prefill
// and would stay on the runtime op; decode still builds the [1, heads, 1, 6]
// score tensor.
// CHECK-LABEL: func.func @cross
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: tensor<1x2x1x6xf16>
// DECOMPOSE: hip.miopen.softmax
func.func @cross(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                 %k: tensor<1x6x16xf16>, %v: tensor<1x6x16xf16>)
    -> tensor<1x1x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x1x16xf16>,
      tensor<1x6x16xf16>, tensor<1x6x16xf16>)
      outs(%empty : tensor<1x1x16xf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x1x16xf16>
  return %r : tensor<1x1x16xf16>
}

// -----

// A scale of 0 means 1/sqrt(head_dim). Head dim 8 prints as this f16.
// Query sequence 1 so the f16 prefill gate does not keep the op.
// CHECK-LABEL: func.func @default_scale
// DECOMPOSE: dense<3.535160e-01>
// DECOMPOSE: hip.miopen.softmax
func.func @default_scale(%ctx: !hip.context, %q: tensor<1x1x16xf16>,
                         %k: tensor<1x1x16xf16>, %v: tensor<1x1x16xf16>)
    -> tensor<1x1x16xf16> {
  %empty = tensor.empty() : tensor<1x1x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x1x16xf16>,
      tensor<1x1x16xf16>, tensor<1x1x16xf16>)
      outs(%empty : tensor<1x1x16xf16>)
      {num_heads = 2 : i64, scale = 0.000000e+00 : f32} : tensor<1x1x16xf16>
  return %r : tensor<1x1x16xf16>
}

// -----

// f32 is not the flash kernel, so a short prefill still splits. Named
// main_graph so hip-fuse-rocmlir outlines the two matmuls.
// CHECK-LABEL: func.func @main_graph({{.*}}tensor<1x4x16xf32>
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: hip.matmul
// DECOMPOSE: hip.mul
// DECOMPOSE: hip.miopen.softmax
// DECOMPOSE: hip.matmul
// FUSE: hip.rocmlir
// FUSE: hip.miopen.softmax
// FUSE: hip.rocmlir
// FUSE-NOT: hip.multi_head_attention
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x4x16xf32>,
               %k: tensor<1x4x16xf32>, %v: tensor<1x4x16xf32>)
    -> tensor<1x4x16xf32> {
  %empty = tensor.empty() : tensor<1x4x16xf32>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf32>,
      tensor<1x4x16xf32>, tensor<1x4x16xf32>)
      outs(%empty : tensor<1x4x16xf32>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x4x16xf32>
  return %r : tensor<1x4x16xf32>
}

// -----

// Padded head dim 272 is above the flash kernel's limit, so this f16 prefill
// still splits. Named main_graph so hip-fuse-rocmlir outlines it.
// CHECK-LABEL: func.func @main_graph({{.*}}tensor<1x2x272xf16>
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: hip.matmul
// DECOMPOSE: hip.miopen.softmax
// FUSE: hip.rocmlir
// FUSE-NOT: hip.multi_head_attention
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x272xf16>,
                     %k: tensor<1x2x272xf16>, %v: tensor<1x2x272xf16>)
    -> tensor<1x2x272xf16> {
  %empty = tensor.empty() : tensor<1x2x272xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x2x272xf16>,
      tensor<1x2x272xf16>, tensor<1x2x272xf16>)
      outs(%empty : tensor<1x2x272xf16>)
      {num_heads = 1 : i64, scale = 5.000000e-01 : f32} : tensor<1x2x272xf16>
  return %r : tensor<1x2x272xf16>
}

// -----

// f64 has no HIP-to-TOSA matmul, so it stays on the runtime kernel.
// CHECK-LABEL: func.func @f64
// CHECK: hip.multi_head_attention
// CHECK-NOT: hip.matmul
func.func @f64(%ctx: !hip.context, %q: tensor<1x4x16xf64>,
               %k: tensor<1x4x16xf64>, %v: tensor<1x4x16xf64>)
    -> tensor<1x4x16xf64> {
  %empty = tensor.empty() : tensor<1x4x16xf64>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf64>,
      tensor<1x4x16xf64>, tensor<1x4x16xf64>)
      outs(%empty : tensor<1x4x16xf64>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x4x16xf64>
  return %r : tensor<1x4x16xf64>
}

// -----

// Causal masking stays on the runtime kernel.
// CHECK-LABEL: func.func @causal
// CHECK: hip.multi_head_attention
func.func @causal(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                  %k: tensor<1x4x16xf16>, %v: tensor<1x4x16xf16>)
    -> tensor<1x4x16xf16> {
  %empty = tensor.empty() : tensor<1x4x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>,
      tensor<1x4x16xf16>, tensor<1x4x16xf16>)
      outs(%empty : tensor<1x4x16xf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32, unidirectional = 1 : i64}
      : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
}
