// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-decompose-multi-head-attention %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,DECOMPOSE
// RUN: hip-mlir-opt --split-input-file --hip-decompose-multi-head-attention \
// RUN:   --hip-fuse-rocmlir %s | hip-mlir-opt --split-input-file \
// RUN:   | FileCheck %s --check-prefixes=CHECK,FUSE

// CHECK-LABEL: func.func @main_graph
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: hip.matmul
// DECOMPOSE: hip.mul
// DECOMPOSE: hip.miopen.softmax
// DECOMPOSE: hip.matmul
// FUSE: hip.rocmlir
// FUSE: hip.miopen.softmax
// FUSE: hip.rocmlir
// FUSE-NOT: hip.multi_head_attention
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

// Cross attention: key sequence 6, query sequence 4.
// CHECK-LABEL: func.func @cross
// DECOMPOSE-NOT: hip.multi_head_attention
// DECOMPOSE: tensor<1x2x4x6xf16>
// DECOMPOSE: hip.miopen.softmax
func.func @cross(%ctx: !hip.context, %q: tensor<1x4x16xf16>,
                 %k: tensor<1x6x16xf16>, %v: tensor<1x6x16xf16>)
    -> tensor<1x4x16xf16> {
  %empty = tensor.empty() : tensor<1x4x16xf16>
  %r = hip.multi_head_attention(%ctx) ins(%q, %k, %v : tensor<1x4x16xf16>,
      tensor<1x6x16xf16>, tensor<1x6x16xf16>)
      outs(%empty : tensor<1x4x16xf16>)
      {num_heads = 2 : i64, scale = 5.000000e-01 : f32} : tensor<1x4x16xf16>
  return %r : tensor<1x4x16xf16>
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
