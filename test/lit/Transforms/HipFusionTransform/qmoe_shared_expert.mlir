// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Qwen3.5-MoE adds a sigmoid-gated shared expert to the routed QMoE:
//
//   y = qmoe(x) + sigmoid(x @ Wl) * down(swiglu(x @ Wg, x @ Wu))
//
// hip-fusion-transform folds the shared expert into the hip.qmoe, keeping
// only the gate-logit projection, so the runtime can run it as one more
// expert slot.

// CHECK-LABEL: func.func @qmoe_shared_expert
// CHECK-SAME: (%[[CTX:[^:]*]]: !hip.context, %[[X:[^:]*]]: tensor<1x?x64xf16>, %[[R:[^:]*]]: tensor<?x4xf16>,
// CHECK-SAME: %[[WG:[^:]*]]: tensor<32x2x16xui8>, %[[SG:[^:]*]]: tensor<32x2xf16>,
// CHECK-SAME: %[[WU:[^:]*]]: tensor<32x2x16xui8>, %[[SU:[^:]*]]: tensor<32x2xf16>,
// CHECK-SAME: %[[WD:[^:]*]]: tensor<64x1x16xui8>, %[[SD:[^:]*]]: tensor<64x1xf16>,
// CHECK-SAME: %[[WL:[^:]*]]: tensor<1x2x16xui8>, %[[SL:[^:]*]]: tensor<1x2xf16>
// CHECK-NOT: hip.swiglu
// CHECK: %[[L:.*]] = hip.matmul_nbits(%[[CTX]]) ins(%[[X]], %[[WL]], %[[SL]] :
// CHECK: %[[D:.*]] = tensor.dim %[[X]], %c1
// CHECK-NOT: hip.swiglu
// CHECK: %[[Y:.*]] = hip.qmoe(%[[CTX]]) ins(%[[X]], %[[R]],
// CHECK-SAME: shared_expert(%[[WG]], %[[SG]], %[[WU]], %[[SU]], %[[WD]], %[[SD]], %[[L]] :
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.add
// CHECK: return %[[Y]], %[[D]]
func.func @qmoe_shared_expert(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %router: tensor<?x4xf16>,
    %wg: tensor<32x2x16xui8>, %sg: tensor<32x2xf16>,
    %wu: tensor<32x2x16xui8>, %su: tensor<32x2xf16>,
    %wd: tensor<64x1x16xui8>, %sd: tensor<64x1xf16>,
    %wl: tensor<1x2x16xui8>, %sl: tensor<1x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>)
    -> (tensor<1x?x64xf16>, index) {
  %c1 = arith.constant 1 : index
  %t = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e0 = tensor.empty(%t) : tensor<1x?x64xf16>
  %moe = hip.qmoe(%ctx) ins(%x, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x?x64xf16>, tensor<?x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e0 : tensor<1x?x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x?x64xf16>
  %e1 = tensor.empty(%t) : tensor<1x?x32xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %wg, %sg : tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x32xf16>
  %e2 = tensor.empty(%t) : tensor<1x?x32xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %wu, %su : tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x32xf16>
  %ta = tensor.dim %g, %c1 : tensor<1x?x32xf16>
  %e3 = tensor.empty(%ta) : tensor<1x?x32xf16>
  %a = hip.swiglu(%ctx) ins(%g, %u : tensor<1x?x32xf16>, tensor<1x?x32xf16>)
      outs(%e3 : tensor<1x?x32xf16>) : tensor<1x?x32xf16>
  %e4 = tensor.empty(%t) : tensor<1x?x64xf16>
  %d = hip.matmul_nbits(%ctx) ins(%a, %wd, %sd : tensor<1x?x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%e4 : tensor<1x?x64xf16>)
      {K = 32 : i64, N = 64 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x64xf16>
  %e5 = tensor.empty(%t) : tensor<1x?x1xf16>
  %l = hip.matmul_nbits(%ctx) ins(%x, %wl, %sl : tensor<1x?x64xf16>, tensor<1x2x16xui8>, tensor<1x2xf16>)
      outs(%e5 : tensor<1x?x1xf16>)
      {K = 64 : i64, N = 1 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x?x1xf16>
  %e6 = tensor.empty(%t) : tensor<1x?x1xf16>
  %s = hip.sigmoid(%ctx) ins(%l : tensor<1x?x1xf16>) outs(%e6 : tensor<1x?x1xf16>) : tensor<1x?x1xf16>
  %e7 = tensor.empty(%t) : tensor<1x?x64xf16>
  %sd2 = hip.mul(%ctx) ins(%d, %s : tensor<1x?x64xf16>, tensor<1x?x1xf16>)
      outs(%e7 : tensor<1x?x64xf16>) -> tensor<1x?x64xf16>
  %tm = tensor.dim %moe, %c1 : tensor<1x?x64xf16>
  %e8 = tensor.empty(%tm) : tensor<1x?x64xf16>
  %y = hip.add(%ctx) ins(%moe, %sd2 : tensor<1x?x64xf16>, tensor<1x?x64xf16>)
      outs(%e8 : tensor<1x?x64xf16>) -> tensor<1x?x64xf16>
  return %y, %tm : tensor<1x?x64xf16>, index
}

// -----

// Commuted add and multiply operands, with the SwiGLU still in primitive
// sigmoid/mul form: the SwiGLU fusion runs first, then this one.
// CHECK-LABEL: func.func @qmoe_shared_expert_commuted
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.add
// CHECK: hip.qmoe
// CHECK-SAME: shared_expert(
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK-NOT: hip.add
// CHECK: return
func.func @qmoe_shared_expert_commuted(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %router: tensor<1x4xf16>,
    %wg: tensor<32x2x16xui8>, %sg: tensor<32x2xf16>,
    %wu: tensor<32x2x16xui8>, %su: tensor<32x2xf16>,
    %wd: tensor<64x1x16xui8>, %sd: tensor<64x1xf16>,
    %wl: tensor<1x2x16xui8>, %sl: tensor<1x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>) -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %moe = hip.qmoe(%ctx) ins(%x, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x1x64xf16>, tensor<1x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %wg, %sg : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %wu, %su : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %gs = hip.sigmoid(%ctx) ins(%g : tensor<1x1x32xf16>) outs(%e3 : tensor<1x1x32xf16>) : tensor<1x1x32xf16>
  %e4 = tensor.empty() : tensor<1x1x32xf16>
  %silu = hip.mul(%ctx) ins(%g, %gs : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
      outs(%e4 : tensor<1x1x32xf16>) -> tensor<1x1x32xf16>
  %e5 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.mul(%ctx) ins(%silu, %u : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
      outs(%e5 : tensor<1x1x32xf16>) -> tensor<1x1x32xf16>
  %e6 = tensor.empty() : tensor<1x1x64xf16>
  %d = hip.matmul_nbits(%ctx) ins(%a, %wd, %sd : tensor<1x1x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%e6 : tensor<1x1x64xf16>)
      {K = 32 : i64, N = 64 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x64xf16>
  %e7 = tensor.empty() : tensor<1x1x1xf16>
  %l = hip.matmul_nbits(%ctx) ins(%x, %wl, %sl : tensor<1x1x64xf16>, tensor<1x2x16xui8>, tensor<1x2xf16>)
      outs(%e7 : tensor<1x1x1xf16>)
      {K = 64 : i64, N = 1 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x1xf16>
  %e8 = tensor.empty() : tensor<1x1x1xf16>
  %s = hip.sigmoid(%ctx) ins(%l : tensor<1x1x1xf16>) outs(%e8 : tensor<1x1x1xf16>) : tensor<1x1x1xf16>
  %e9 = tensor.empty() : tensor<1x1x64xf16>
  %sd2 = hip.mul(%ctx) ins(%s, %d : tensor<1x1x1xf16>, tensor<1x1x64xf16>)
      outs(%e9 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e10 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.add(%ctx) ins(%sd2, %moe : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%e10 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}

// -----

// The shared expert reads another activation than the routed experts: the
// runtime has only one input to project, so nothing fuses.
// CHECK-LABEL: func.func @qmoe_shared_other_input
// CHECK: hip.qmoe
// CHECK-NOT: shared_expert(
// CHECK: hip.swiglu
// CHECK: hip.add
func.func @qmoe_shared_other_input(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %x2: tensor<1x1x64xf16>, %router: tensor<1x4xf16>,
    %wg: tensor<32x2x16xui8>, %sg: tensor<32x2xf16>,
    %wu: tensor<32x2x16xui8>, %su: tensor<32x2xf16>,
    %wd: tensor<64x1x16xui8>, %sd: tensor<64x1xf16>,
    %wl: tensor<1x2x16xui8>, %sl: tensor<1x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>) -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %moe = hip.qmoe(%ctx) ins(%x, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x1x64xf16>, tensor<1x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x2, %wg, %sg : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x2, %wu, %su : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.swiglu(%ctx) ins(%g, %u : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
      outs(%e3 : tensor<1x1x32xf16>) : tensor<1x1x32xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %d = hip.matmul_nbits(%ctx) ins(%a, %wd, %sd : tensor<1x1x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%e4 : tensor<1x1x64xf16>)
      {K = 32 : i64, N = 64 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x64xf16>
  %e5 = tensor.empty() : tensor<1x1x1xf16>
  %l = hip.matmul_nbits(%ctx) ins(%x, %wl, %sl : tensor<1x1x64xf16>, tensor<1x2x16xui8>, tensor<1x2xf16>)
      outs(%e5 : tensor<1x1x1xf16>)
      {K = 64 : i64, N = 1 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x1xf16>
  %e6 = tensor.empty() : tensor<1x1x1xf16>
  %s = hip.sigmoid(%ctx) ins(%l : tensor<1x1x1xf16>) outs(%e6 : tensor<1x1x1xf16>) : tensor<1x1x1xf16>
  %e7 = tensor.empty() : tensor<1x1x64xf16>
  %sd2 = hip.mul(%ctx) ins(%d, %s : tensor<1x1x64xf16>, tensor<1x1x1xf16>)
      outs(%e7 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e8 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.add(%ctx) ins(%moe, %sd2 : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%e8 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}

// -----

// The down projection is also returned, so erasing it would change the
// graph's outputs.
// CHECK-LABEL: func.func @qmoe_shared_down_multi_use
// CHECK: hip.qmoe
// CHECK-NOT: shared_expert(
// CHECK: hip.swiglu
// CHECK: hip.add
func.func @qmoe_shared_down_multi_use(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %router: tensor<1x4xf16>,
    %wg: tensor<32x2x16xui8>, %sg: tensor<32x2xf16>,
    %wu: tensor<32x2x16xui8>, %su: tensor<32x2xf16>,
    %wd: tensor<64x1x16xui8>, %sd: tensor<64x1xf16>,
    %wl: tensor<1x2x16xui8>, %sl: tensor<1x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %moe = hip.qmoe(%ctx) ins(%x, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x1x64xf16>, tensor<1x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %wg, %sg : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %wu, %su : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.swiglu(%ctx) ins(%g, %u : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
      outs(%e3 : tensor<1x1x32xf16>) : tensor<1x1x32xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %d = hip.matmul_nbits(%ctx) ins(%a, %wd, %sd : tensor<1x1x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%e4 : tensor<1x1x64xf16>)
      {K = 32 : i64, N = 64 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x64xf16>
  %e5 = tensor.empty() : tensor<1x1x1xf16>
  %l = hip.matmul_nbits(%ctx) ins(%x, %wl, %sl : tensor<1x1x64xf16>, tensor<1x2x16xui8>, tensor<1x2xf16>)
      outs(%e5 : tensor<1x1x1xf16>)
      {K = 64 : i64, N = 1 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x1xf16>
  %e6 = tensor.empty() : tensor<1x1x1xf16>
  %s = hip.sigmoid(%ctx) ins(%l : tensor<1x1x1xf16>) outs(%e6 : tensor<1x1x1xf16>) : tensor<1x1x1xf16>
  %e7 = tensor.empty() : tensor<1x1x64xf16>
  %sd2 = hip.mul(%ctx) ins(%d, %s : tensor<1x1x64xf16>, tensor<1x1x1xf16>)
      outs(%e7 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e8 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.add(%ctx) ins(%moe, %sd2 : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%e8 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y, %d : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// The shared branch is scaled by the raw logit, not its sigmoid.
// CHECK-LABEL: func.func @qmoe_shared_ungated
// CHECK: hip.qmoe
// CHECK-NOT: shared_expert(
// CHECK: hip.swiglu
// CHECK: hip.add
func.func @qmoe_shared_ungated(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
    %router: tensor<1x4xf16>,
    %wg: tensor<32x2x16xui8>, %sg: tensor<32x2xf16>,
    %wu: tensor<32x2x16xui8>, %su: tensor<32x2xf16>,
    %wd: tensor<64x1x16xui8>, %sd: tensor<64x1xf16>,
    %wl: tensor<1x2x16xui8>, %sl: tensor<1x2xf16>,
    %fc1_w: tensor<4x64x32xui8>, %fc1_s: tensor<4x64x2xf16>,
    %fc2_w: tensor<4x64x16xui8>, %fc2_s: tensor<4x64x1xf16>) -> tensor<1x1x64xf16> {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %moe = hip.qmoe(%ctx) ins(%x, %router, %fc1_w, %fc1_s, %fc2_w, %fc2_s :
      tensor<1x1x64xf16>, tensor<1x4xf16>, tensor<4x64x32xui8>,
      tensor<4x64x2xf16>, tensor<4x64x16xui8>, tensor<4x64x1xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {activation_alpha = 1.000000e+00 : f32, activation_beta = 0.000000e+00 : f32,
       activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 4 : i64,
       k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64,
       swiglu_limit = 0x7F800000 : f32, use_sparse_mixer = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %wg, %sg : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %wu, %su : tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.swiglu(%ctx) ins(%g, %u : tensor<1x1x32xf16>, tensor<1x1x32xf16>)
      outs(%e3 : tensor<1x1x32xf16>) : tensor<1x1x32xf16>
  %e4 = tensor.empty() : tensor<1x1x64xf16>
  %d = hip.matmul_nbits(%ctx) ins(%a, %wd, %sd : tensor<1x1x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%e4 : tensor<1x1x64xf16>)
      {K = 32 : i64, N = 64 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x64xf16>
  %e5 = tensor.empty() : tensor<1x1x1xf16>
  %l = hip.matmul_nbits(%ctx) ins(%x, %wl, %sl : tensor<1x1x64xf16>, tensor<1x2x16xui8>, tensor<1x2xf16>)
      outs(%e5 : tensor<1x1x1xf16>)
      {K = 64 : i64, N = 1 : i64, accuracy_level = 4 : i64, bits = 4 : i64, block_size = 32 : i64, zp_elem_size = 2 : i64} : tensor<1x1x1xf16>
  %e7 = tensor.empty() : tensor<1x1x64xf16>
  %sd2 = hip.mul(%ctx) ins(%d, %l : tensor<1x1x64xf16>, tensor<1x1x1xf16>)
      outs(%e7 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  %e8 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.add(%ctx) ins(%moe, %sd2 : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%e8 : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}
