// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The pre-mix gate path of a hyper-connection boundary (column split of the
// projection, scalar mul, Swish, 4-bit projection, sigmoid feature pre-mix)
// becomes one hip.hyper_connection_mixer; a later read of the trailing gate
// columns becomes its second result.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_static
// CHECK: %[[P:.*]] = hip.matmul_nbits
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.swish
// CHECK: %[[R:.*]]:4 = hip.hyper_connection_mixer(%{{[a-z0-9_]+}}) ins(%{{[a-z0-9_]+}}, %[[P]],
// CHECK-SAME: act_scale = 2.500000e-01 : f32, act_width = 64 : i64, block_size = 32 : i64, branches = 2 : i64, hidden = 64 : i64, reduction_scale = 5.000000e-01 : f32, swish_alpha = 1.000000e+00 : f32
// CHECK-NOT: hip.hyper_connection_pre_mix
// CHECK-NOT: hip.matmul_nbits
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @fuse_static(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                       %streams: tensor<1x1x128xf16>,
                       %B0: tensor<66x2x16xui8>, %S0: tensor<66x2xf16>,
                       %B1: tensor<128x2x16xui8>, %S1: tensor<128x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x2xf16>) {
  %e0 = tensor.empty() : tensor<1x1x66xf16>
  %p = hip.matmul_nbits(%ctx) ins(%x, %B0, %S0 : tensor<1x1x64xf16>,
      tensor<66x2x16xui8>, tensor<66x2xf16>) outs(%e0 : tensor<1x1x66xf16>)
      {K = 64 : i64, N = 66 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x66xf16>
  %a = tensor.extract_slice %p[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : tensor<1x1x66xf16> to tensor<1x1x64xf16>
  %g = tensor.extract_slice %p[0, 0, 64] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x66xf16> to tensor<1x1x2xf16>
  %c = hip.constant {value = dense<2.500000e-01> : tensor<f16>} : tensor<f16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %s = hip.mul(%ctx) ins(%a, %c : tensor<1x1x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %w = hip.swish(%ctx) ins(%s : tensor<1x1x64xf16>)
      outs(%e2 : tensor<1x1x64xf16>) {alpha = 1.000000e+00 : f64}
      : tensor<1x1x64xf16>
  %e3 = tensor.empty() : tensor<1x1x128xf16>
  %m = hip.matmul_nbits(%ctx) ins(%w, %B1, %S1 : tensor<1x1x64xf16>,
      tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e3 : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x128xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m
      : tensor<1x1x128xf16>, tensor<1x1x128xf16>) outs(%e4 : tensor<1x1x64xf16>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       gate_sigmoid = 1 : i64, hidden = 64 : i64,
       reduction_scale = 5.000000e-01 : f32} : tensor<1x1x64xf16>
  return %y, %g : tensor<1x1x64xf16>, tensor<1x1x2xf16>
}

// -----

// Dynamic leading axes: the slices size them with tensor.dim of the
// projection, and the gate result gets a fresh destination sized the same way.
// CHECK-LABEL: func.func @fuse_dynamic
// CHECK: %[[P:.*]] = hip.matmul_nbits
// CHECK: tensor.empty({{.*}}) : tensor<?x?x128xf16>
// CHECK: %[[D0:.*]] = tensor.dim %[[P]], %c0
// CHECK-NEXT: %[[D1:.*]] = tensor.dim %[[P]], %c1
// CHECK-NEXT: %[[G:.*]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?x2xf16>
// CHECK-NEXT: %[[R:.*]]:4 = hip.hyper_connection_mixer(%{{[a-z0-9_]+}}) ins(%{{[a-z0-9_]+}}, %[[P]], %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} : {{[^)]*}}) outs(%{{[a-z0-9_]+}}, %[[G]],
// CHECK-NOT: hip.hyper_connection_pre_mix
// CHECK: return %[[R]]#0, %[[R]]#1
func.func @fuse_dynamic(%ctx: !hip.context, %x: tensor<?x?x64xf16>,
                        %streams: tensor<?x?x128xf16>,
                        %B0: tensor<66x2x16xui8>, %S0: tensor<66x2xf16>,
                        %B1: tensor<128x2x16xui8>, %S1: tensor<128x2xf16>)
    -> (tensor<?x?x64xf16>, tensor<?x?x2xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n0 = tensor.dim %x, %c0 : tensor<?x?x64xf16>
  %n1 = tensor.dim %x, %c1 : tensor<?x?x64xf16>
  %e0 = tensor.empty(%n0, %n1) : tensor<?x?x66xf16>
  %p = hip.matmul_nbits(%ctx) ins(%x, %B0, %S0 : tensor<?x?x64xf16>,
      tensor<66x2x16xui8>, tensor<66x2xf16>) outs(%e0 : tensor<?x?x66xf16>)
      {K = 64 : i64, N = 66 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<?x?x66xf16>
  %d0 = tensor.dim %p, %c0 : tensor<?x?x66xf16>
  %d1 = tensor.dim %p, %c1 : tensor<?x?x66xf16>
  %a = tensor.extract_slice %p[0, 0, 0] [%d0, %d1, 64] [1, 1, 1]
      : tensor<?x?x66xf16> to tensor<?x?x64xf16>
  %g = tensor.extract_slice %p[0, 0, 64] [%d0, %d1, 2] [1, 1, 1]
      : tensor<?x?x66xf16> to tensor<?x?x2xf16>
  %c = hip.constant {value = dense<2.500000e-01> : tensor<f16>} : tensor<f16>
  %e1 = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %s = hip.mul(%ctx) ins(%a, %c : tensor<?x?x64xf16>, tensor<f16>)
      outs(%e1 : tensor<?x?x64xf16>) -> tensor<?x?x64xf16>
  %e2 = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %w = hip.swish(%ctx) ins(%s : tensor<?x?x64xf16>)
      outs(%e2 : tensor<?x?x64xf16>) {alpha = 1.000000e+00 : f64}
      : tensor<?x?x64xf16>
  %e3 = tensor.empty(%d0, %d1) : tensor<?x?x128xf16>
  %m = hip.matmul_nbits(%ctx) ins(%w, %B1, %S1 : tensor<?x?x64xf16>,
      tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e3 : tensor<?x?x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<?x?x128xf16>
  %e4 = tensor.empty(%d0, %d1) : tensor<?x?x64xf16>
  %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m
      : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%e4 : tensor<?x?x64xf16>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       gate_sigmoid = 1 : i64, hidden = 64 : i64,
       reduction_scale = 5.000000e-01 : f32} : tensor<?x?x64xf16>
  return %y, %g : tensor<?x?x64xf16>, tensor<?x?x2xf16>
}

// -----

// The Swish output has a second reader, so the chain stays unfused.
// CHECK-LABEL: func.func @reject_extra_use
// CHECK: hip.swish
// CHECK: hip.hyper_connection_pre_mix
// CHECK-NOT: hip.hyper_connection_mixer
func.func @reject_extra_use(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                            %streams: tensor<1x1x128xf16>,
                            %B0: tensor<66x2x16xui8>, %S0: tensor<66x2xf16>,
                            %B1: tensor<128x2x16xui8>, %S1: tensor<128x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x66xf16>
  %p = hip.matmul_nbits(%ctx) ins(%x, %B0, %S0 : tensor<1x1x64xf16>,
      tensor<66x2x16xui8>, tensor<66x2xf16>) outs(%e0 : tensor<1x1x66xf16>)
      {K = 64 : i64, N = 66 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x66xf16>
  %a = tensor.extract_slice %p[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : tensor<1x1x66xf16> to tensor<1x1x64xf16>
  %c = hip.constant {value = dense<2.500000e-01> : tensor<f16>} : tensor<f16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %s = hip.mul(%ctx) ins(%a, %c : tensor<1x1x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %w = hip.swish(%ctx) ins(%s : tensor<1x1x64xf16>)
      outs(%e2 : tensor<1x1x64xf16>) {alpha = 1.000000e+00 : f64}
      : tensor<1x1x64xf16>
  %e3 = tensor.empty() : tensor<1x1x128xf16>
  %m = hip.matmul_nbits(%ctx) ins(%w, %B1, %S1 : tensor<1x1x64xf16>,
      tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e3 : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x128xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m
      : tensor<1x1x128xf16>, tensor<1x1x128xf16>) outs(%e4 : tensor<1x1x64xf16>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       gate_sigmoid = 1 : i64, hidden = 64 : i64,
       reduction_scale = 5.000000e-01 : f32} : tensor<1x1x64xf16>
  return %y, %w : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// A tensor (not a splat scalar) scales the activation: no fusion.
// CHECK-LABEL: func.func @reject_tensor_scale
// CHECK: hip.mul
// CHECK-NOT: hip.hyper_connection_mixer
func.func @reject_tensor_scale(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                               %streams: tensor<1x1x128xf16>,
                               %scale: tensor<1x1x64xf16>,
                               %B0: tensor<66x2x16xui8>, %S0: tensor<66x2xf16>,
                               %B1: tensor<128x2x16xui8>,
                               %S1: tensor<128x2xf16>) -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x66xf16>
  %p = hip.matmul_nbits(%ctx) ins(%x, %B0, %S0 : tensor<1x1x64xf16>,
      tensor<66x2x16xui8>, tensor<66x2xf16>) outs(%e0 : tensor<1x1x66xf16>)
      {K = 64 : i64, N = 66 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x66xf16>
  %a = tensor.extract_slice %p[0, 0, 0] [1, 1, 64] [1, 1, 1]
      : tensor<1x1x66xf16> to tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %s = hip.mul(%ctx) ins(%a, %scale : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%e1 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %w = hip.swish(%ctx) ins(%s : tensor<1x1x64xf16>)
      outs(%e2 : tensor<1x1x64xf16>) {alpha = 1.000000e+00 : f64}
      : tensor<1x1x64xf16>
  %e3 = tensor.empty() : tensor<1x1x128xf16>
  %m = hip.matmul_nbits(%ctx) ins(%w, %B1, %S1 : tensor<1x1x64xf16>,
      tensor<128x2x16xui8>, tensor<128x2xf16>) outs(%e3 : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x128xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m
      : tensor<1x1x128xf16>, tensor<1x1x128xf16>) outs(%e4 : tensor<1x1x64xf16>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       gate_sigmoid = 1 : i64, hidden = 64 : i64,
       reduction_scale = 5.000000e-01 : f32} : tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}
