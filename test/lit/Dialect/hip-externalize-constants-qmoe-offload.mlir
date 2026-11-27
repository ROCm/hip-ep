// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: env -u HIPDNN_EP_QMOE_EXPERT_OFFLOAD hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %t/experts.mlir | FileCheck %s
// RUN: env HIPDNN_EP_QMOE_EXPERT_OFFLOAD=0 hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %t/experts.mlir | FileCheck %s --check-prefix=OFF
// RUN: env -u HIPDNN_EP_QMOE_EXPERT_OFFLOAD hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %t/mixed.mlir | FileCheck %s --check-prefix=MIXED

// CHECK: module attributes
// CHECK-SAME: hipdnn.constant_source_kinds = array<i32: 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>
// OFF: module attributes
// OFF-SAME: hipdnn.constant_source_kinds = array<i32: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>
// MIXED: module attributes
// MIXED-SAME: hipdnn.constant_source_kinds = array<i32: 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>

//--- experts.mlir
// @b has zero points, and @c also returns its fc1 weights, so both keep all
// of their experts on the GPU.
func.func @experts(%ctx: !hip.context, %x: tensor<1x2x64xf16>,
                   %r: tensor<2x4xf16>, %zp1: tensor<4x64x1xui8>,
                   %zp2: tensor<4x64x1xui8>, %out: tensor<1x2x64xf16>)
    -> (tensor<1x2x64xf16>, tensor<1x2x64xf16>, tensor<1x2x64xf16>,
        tensor<4x64x16xui8>) {
  %a1w = hip.constant {location = "/m/w.data", offset = 0 : i64, size = 4096 : i64} : tensor<4x64x16xui8>
  %a1s = hip.constant {location = "/m/w.data", offset = 4096 : i64, size = 1024 : i64} : tensor<4x64x2xf16>
  %a1b = hip.constant {location = "/m/w.data", offset = 5120 : i64, size = 512 : i64} : tensor<4x64xf16>
  %a2w = hip.constant {location = "/m/w.data", offset = 5632 : i64, size = 2048 : i64} : tensor<4x64x8xui8>
  %a2s = hip.constant {location = "/m/w.data", offset = 7680 : i64, size = 512 : i64} : tensor<4x64x1xf16>
  %a2b = hip.constant {location = "/m/w.data", offset = 8192 : i64, size = 512 : i64} : tensor<4x64xf16>
  %a = hip.qmoe(%ctx) ins(%x, %r, %a1w, %a1s, %a2w, %a2s : tensor<1x2x64xf16>, tensor<2x4xf16>, tensor<4x64x16xui8>, tensor<4x64x2xf16>, tensor<4x64x8xui8>, tensor<4x64x1xf16>)
      fc1_bias(%a1b : tensor<4x64xf16>) fc2_bias(%a2b : tensor<4x64xf16>)
      outs(%out : tensor<1x2x64xf16>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64} : tensor<1x2x64xf16>

  %b1w = hip.constant {location = "/m/w.data", offset = 8704 : i64, size = 4096 : i64} : tensor<4x64x16xui8>
  %b1s = hip.constant {location = "/m/w.data", offset = 12800 : i64, size = 1024 : i64} : tensor<4x64x2xf16>
  %b1b = hip.constant {location = "/m/w.data", offset = 13824 : i64, size = 512 : i64} : tensor<4x64xf16>
  %b2w = hip.constant {location = "/m/w.data", offset = 14336 : i64, size = 2048 : i64} : tensor<4x64x8xui8>
  %b2s = hip.constant {location = "/m/w.data", offset = 16384 : i64, size = 512 : i64} : tensor<4x64x1xf16>
  %b2b = hip.constant {location = "/m/w.data", offset = 16896 : i64, size = 512 : i64} : tensor<4x64xf16>
  %b = hip.qmoe(%ctx) ins(%x, %r, %b1w, %b1s, %b2w, %b2s : tensor<1x2x64xf16>, tensor<2x4xf16>, tensor<4x64x16xui8>, tensor<4x64x2xf16>, tensor<4x64x8xui8>, tensor<4x64x1xf16>)
      fc1_bias(%b1b : tensor<4x64xf16>) fc2_bias(%b2b : tensor<4x64xf16>)
      fc1_zero_points(%zp1 : tensor<4x64x1xui8>) fc2_zero_points(%zp2 : tensor<4x64x1xui8>)
      outs(%out : tensor<1x2x64xf16>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64} : tensor<1x2x64xf16>

  %c1w = hip.constant {location = "/m/w.data", offset = 17408 : i64, size = 4096 : i64} : tensor<4x64x16xui8>
  %c1s = hip.constant {location = "/m/w.data", offset = 21504 : i64, size = 1024 : i64} : tensor<4x64x2xf16>
  %c1b = hip.constant {location = "/m/w.data", offset = 22528 : i64, size = 512 : i64} : tensor<4x64xf16>
  %c2w = hip.constant {location = "/m/w.data", offset = 23040 : i64, size = 2048 : i64} : tensor<4x64x8xui8>
  %c2s = hip.constant {location = "/m/w.data", offset = 25088 : i64, size = 512 : i64} : tensor<4x64x1xf16>
  %c2b = hip.constant {location = "/m/w.data", offset = 25600 : i64, size = 512 : i64} : tensor<4x64xf16>
  %c = hip.qmoe(%ctx) ins(%x, %r, %c1w, %c1s, %c2w, %c2s : tensor<1x2x64xf16>, tensor<2x4xf16>, tensor<4x64x16xui8>, tensor<4x64x2xf16>, tensor<4x64x8xui8>, tensor<4x64x1xf16>)
      fc1_bias(%c1b : tensor<4x64xf16>) fc2_bias(%c2b : tensor<4x64xf16>)
      outs(%out : tensor<1x2x64xf16>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64} : tensor<1x2x64xf16>
  return %a, %b, %c, %c1w : tensor<1x2x64xf16>, tensor<1x2x64xf16>, tensor<1x2x64xf16>, tensor<4x64x16xui8>
}

//--- mixed.mlir
// The two QMoEs have different expert counts, so neither is offloaded.
func.func @mixed(%ctx: !hip.context, %x: tensor<1x2x64xf16>,
                 %r4: tensor<2x4xf16>, %r8: tensor<2x8xf16>,
                 %out: tensor<1x2x64xf16>)
    -> (tensor<1x2x64xf16>, tensor<1x2x64xf16>) {
  %a1w = hip.constant {location = "/m/w.data", offset = 0 : i64, size = 4096 : i64} : tensor<4x64x16xui8>
  %a1s = hip.constant {location = "/m/w.data", offset = 4096 : i64, size = 1024 : i64} : tensor<4x64x2xf16>
  %a1b = hip.constant {location = "/m/w.data", offset = 5120 : i64, size = 512 : i64} : tensor<4x64xf16>
  %a2w = hip.constant {location = "/m/w.data", offset = 5632 : i64, size = 2048 : i64} : tensor<4x64x8xui8>
  %a2s = hip.constant {location = "/m/w.data", offset = 7680 : i64, size = 512 : i64} : tensor<4x64x1xf16>
  %a2b = hip.constant {location = "/m/w.data", offset = 8192 : i64, size = 512 : i64} : tensor<4x64xf16>
  %a = hip.qmoe(%ctx) ins(%x, %r4, %a1w, %a1s, %a2w, %a2s : tensor<1x2x64xf16>, tensor<2x4xf16>, tensor<4x64x16xui8>, tensor<4x64x2xf16>, tensor<4x64x8xui8>, tensor<4x64x1xf16>)
      fc1_bias(%a1b : tensor<4x64xf16>) fc2_bias(%a2b : tensor<4x64xf16>)
      outs(%out : tensor<1x2x64xf16>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64} : tensor<1x2x64xf16>
  %b1w = hip.constant {location = "/m/w.data", offset = 8704 : i64, size = 8192 : i64} : tensor<8x64x16xui8>
  %b1s = hip.constant {location = "/m/w.data", offset = 16896 : i64, size = 2048 : i64} : tensor<8x64x2xf16>
  %b1b = hip.constant {location = "/m/w.data", offset = 18944 : i64, size = 1024 : i64} : tensor<8x64xf16>
  %b2w = hip.constant {location = "/m/w.data", offset = 19968 : i64, size = 4096 : i64} : tensor<8x64x8xui8>
  %b2s = hip.constant {location = "/m/w.data", offset = 24064 : i64, size = 1024 : i64} : tensor<8x64x1xf16>
  %b2b = hip.constant {location = "/m/w.data", offset = 25088 : i64, size = 1024 : i64} : tensor<8x64xf16>
  %b = hip.qmoe(%ctx) ins(%x, %r8, %b1w, %b1s, %b2w, %b2s : tensor<1x2x64xf16>, tensor<2x8xf16>, tensor<8x64x16xui8>, tensor<8x64x2xf16>, tensor<8x64x8xui8>, tensor<8x64x1xf16>)
      fc1_bias(%b1b : tensor<8x64xf16>) fc2_bias(%b2b : tensor<8x64xf16>)
      outs(%out : tensor<1x2x64xf16>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64} : tensor<1x2x64xf16>
  return %a, %b : tensor<1x2x64xf16>, tensor<1x2x64xf16>
}
