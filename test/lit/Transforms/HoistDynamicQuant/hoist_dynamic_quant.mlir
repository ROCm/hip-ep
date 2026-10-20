// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --hip-hoist-dynamic-quant gives the int4 consumers of an activation
// one shared int8 quantization: written by the producing hip.skip_rms_norm
// when there is one, otherwise by one hip.dynamic_quant_i8 for two or more
// consumers, and not at all for a single consumer or ineligible ops.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-hoist-dynamic-quant --split-input-file | FileCheck %s

// The norm writes the quantization; the group and the MoE both read it.
// CHECK-LABEL: func.func @norm_producer
// CHECK: %[[EQ:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x64xi8>
// CHECK: %[[ES:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x2xf32>
// CHECK: %[[N:.*]]:4 = hip.skip_rms_norm(%{{[^)]*}}) ins(
// CHECK-SAME: outs(%{{[^,]*}}, %{{[^,]*}}, %[[EQ]], %[[ES]] :
// CHECK-SAME: quant_block_size = 32 : i64
// CHECK: hip.matmul_nbits_group(%{{.*}}) ins(%[[N]]#0 : tensor<1x?x64xf16>)
// CHECK-SAME: a_quant(%[[N]]#2, %[[N]]#3 : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK: hip.qmoe(%{{.*}}) ins(%[[N]]#0,
// CHECK-SAME: input_quant(%[[N]]#2, %[[N]]#3 : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK-NOT: hip.dynamic_quant_i8
func.func @norm_producer(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %r: tensor<1x?x64xf16>, %gamma: tensor<64xf16>, %router: tensor<?x4xf16>,
    %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>,
    %w1: tensor<16x2x16xui8>, %s1: tensor<16x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>)
    -> (tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x16xf16>,
        tensor<1x?x64xf16>) {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e0 = tensor.empty(%t) : tensor<1x?x64xf16>
  %e1 = tensor.empty(%t) : tensor<1x?x64xf16>
  %n:2 = hip.skip_rms_norm(%ctx) ins(%x, %r, %gamma :
      tensor<1x?x64xf16>, tensor<1x?x64xf16>, tensor<64xf16>)
      outs(%e0, %e1 : tensor<1x?x64xf16>, tensor<1x?x64xf16>)
      {epsilon = 9.99999997E-7 : f32} : tensor<1x?x64xf16>, tensor<1x?x64xf16>
  %e2 = tensor.empty(%t) : tensor<1x?x32xf16>
  %e3 = tensor.empty(%t) : tensor<1x?x16xf16>
  %p:2 = hip.matmul_nbits_group(%ctx) ins(%n#0 : tensor<1x?x64xf16>)
      weights(%w0, %w1 : tensor<32x2x16xui8>, tensor<16x2x16xui8>)
      scales(%s0, %s1 : tensor<32x2xf16>, tensor<16x2xf16>)
      outs(%e2, %e3 : tensor<1x?x32xf16>, tensor<1x?x16xf16>)
      {K = 64 : i64, Ns = array<i64: 32, 16>, accuracy_level = 4 : i64,
       bits = 4 : i64, block_size = 32 : i64}
      : tensor<1x?x32xf16>, tensor<1x?x16xf16>
  %e4 = tensor.empty(%t) : tensor<1x?x64xf16>
  %moe = hip.qmoe(%ctx) ins(%n#0, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x?x64xf16>, tensor<?x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e4 : tensor<1x?x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x?x64xf16>
  return %n#1, %p#0, %p#1, %moe
      : tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x16xf16>,
        tensor<1x?x64xf16>
}

// -----

// No norm to fold into: two consumers share one hip.dynamic_quant_i8 placed
// right after the activation's definition.
// CHECK-LABEL: func.func @two_consumers
// CHECK-SAME: %[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x1x64xf16>
// CHECK: %[[Q:.*]]:2 = hip.dynamic_quant_i8(%[[CTX]]) ins(%[[X]] : tensor<1x1x64xf16>)
// CHECK-SAME: {block_size = 32 : i64} : tensor<1x1x64xi8>, tensor<1x1x2xf32>
// CHECK: hip.matmul_nbits(%[[CTX]]) ins(%[[X]], {{.*}}) a_quant(%[[Q]]#0, %[[Q]]#1 :
// CHECK: hip.matmul_nbits(%[[CTX]]) ins(%[[X]], {{.*}}) a_quant(%[[Q]]#0, %[[Q]]#1 :
func.func @two_consumers(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>,
    %w1: tensor<16x2x16xui8>, %s1: tensor<16x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x16xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %w0, %s0 : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x16xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %w1, %s1 : tensor<1x1x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e1 : tensor<1x1x16xf16>)
      {K = 64 : i64, N = 16 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x16xf16>
  return %y0, %y1 : tensor<1x1x32xf16>, tensor<1x1x16xf16>
}

// -----

// A gated norm writes the quantization of its output for even a single
// consumer, which then needs no launch of its own.
// CHECK-LABEL: func.func @gated_norm_producer
// CHECK: %[[EQ:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x64xi8>
// CHECK: %[[ES:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x2xf32>
// CHECK: %[[N:.*]]:3 = hip.gated_rms_norm(%{{[^)]*}}) ins(
// CHECK-SAME: outs(%{{[^,]*}}, %[[EQ]], %[[ES]] :
// CHECK-SAME: quant_block_size = 32 : i64
// CHECK: hip.matmul_nbits(%{{.*}}) ins(%[[N]]#0,
// CHECK-SAME: a_quant(%[[N]]#1, %[[N]]#2 : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK-NOT: hip.dynamic_quant_i8
func.func @gated_norm_producer(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %z: tensor<1x?x64xf16>, %w: tensor<32xf16>,
    %w0: tensor<16x2x16xui8>, %s0: tensor<16x2xf16>) -> tensor<1x?x16xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e = tensor.empty(%t) : tensor<1x?x64xf16>
  %n = hip.gated_rms_norm(%ctx) ins(%x, %z, %w : tensor<1x?x64xf16>, tensor<1x?x64xf16>, tensor<32xf16>)
      outs(%e : tensor<1x?x64xf16>) {epsilon = 9.99999997E-7 : f32} : tensor<1x?x64xf16>
  %e0 = tensor.empty(%t) : tensor<1x?x16xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%n, %w0, %s0 : tensor<1x?x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e0 : tensor<1x?x16xf16>)
      {K = 64 : i64, N = 16 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x16xf16>
  return %y0 : tensor<1x?x16xf16>
}

// -----

// So does a sigmoid gate (the output gate of gated attention).
// CHECK-LABEL: func.func @sigmoid_mul_producer
// CHECK: %[[EQ:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x64xi8>
// CHECK: %[[ES:.*]] = tensor.empty(%{{.*}}) : tensor<1x?x2xf32>
// CHECK: %[[Y:.*]]:3 = hip.sigmoid_mul(%{{[^)]*}}) ins(
// CHECK-SAME: outs(%{{[^,]*}}, %[[EQ]], %[[ES]] :
// CHECK-SAME: quant_block_size = 32 : i64
// CHECK: hip.matmul_nbits(%{{.*}}) ins(%[[Y]]#0,
// CHECK-SAME: a_quant(%[[Y]]#1, %[[Y]]#2 : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK-NOT: hip.dynamic_quant_i8
func.func @sigmoid_mul_producer(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %g: tensor<1x?x64xf16>,
    %w0: tensor<16x2x16xui8>, %s0: tensor<16x2xf16>) -> tensor<1x?x16xf16> {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e = tensor.empty(%t) : tensor<1x?x64xf16>
  %y = hip.sigmoid_mul(%ctx) ins(%x, %g : tensor<1x?x64xf16>, tensor<1x?x64xf16>)
      outs(%e : tensor<1x?x64xf16>) : tensor<1x?x64xf16>
  %e0 = tensor.empty(%t) : tensor<1x?x16xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%y, %w0, %s0 : tensor<1x?x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e0 : tensor<1x?x16xf16>)
      {K = 64 : i64, N = 16 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x16xf16>
  return %y0 : tensor<1x?x16xf16>
}

// -----

// A single consumer already quantizes in its own launch, so a separate one
// would not pay.
// CHECK-LABEL: func.func @single_consumer
// CHECK-NOT: hip.dynamic_quant_i8
// CHECK-NOT: a_quant
func.func @single_consumer(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>) -> tensor<1x1x32xf16> {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %w0, %s0 : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  return %y0 : tensor<1x1x32xf16>
}

// -----

// The integer path needs K % 32 == 0 and int4 weights: neither a K = 48 nor
// an 8-bit consumer takes part.
// CHECK-LABEL: func.func @ineligible
// CHECK-NOT: hip.dynamic_quant_i8
// CHECK-NOT: a_quant
func.func @ineligible(%ctx: !hip.context, %x: tensor<1x1x48xf16>,
    %x8: tensor<1x1x64xf16>,
    %w0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>,
    %w8: tensor<32x2x32xui8>, %s8: tensor<32x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>,
        tensor<1x1x32xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %w0, %s0 : tensor<1x1x48xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 48 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %w0, %s0 : tensor<1x1x48xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 48 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x8, %w8, %s8 : tensor<1x1x64xf16>, tensor<32x2x32xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 8 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %y3 = hip.matmul_nbits(%ctx) ins(%x8, %w8, %s8 : tensor<1x1x64xf16>, tensor<32x2x32xui8>, tensor<32x2xf16>)
      outs(%e3 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 8 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  return %y0, %y1, %y2, %y3 : tensor<1x1x32xf16>, tensor<1x1x32xf16>,
      tensor<1x1x32xf16>, tensor<1x1x32xf16>
}
