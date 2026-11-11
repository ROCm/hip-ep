// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --hip-share-matmul-nbits-activation tags hip.matmul_nbits ops that
// read one tensor value in one block: every member gets the same
// module-unique hip.act_quant_group and only the first in program order gets
// hip.act_quant_leader. Ops with a unique input, or whose siblings live in
// another block, stay untagged. An input written by an int8-capable
// hyper-connection producer makes that producer the group's leader.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-share-matmul-nbits-activation | FileCheck %s

module {
  // Three readers of %A form group 1; the reader of %X stays untagged.
  // CHECK-LABEL: func.func @siblings
  func.func @siblings(%ctx: !hip.context, %A: tensor<1x256xf16>,
                      %X: tensor<1x256xf16>, %B: tensor<64x8x16xui8>,
                      %S: tensor<64x8xf16>, %E: tensor<1x64xf16>)
      -> (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>,
          tensor<1x64xf16>) {
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 1 : i64, hip.act_quant_leader, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, zp_elem_size
    %1 = hip.matmul_nbits(%ctx) ins(%X, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 1 : i64, zp_elem_size
    %2 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 1 : i64, zp_elem_size
    %3 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0, %1, %2, %3 : tensor<1x64xf16>, tensor<1x64xf16>,
        tensor<1x64xf16>, tensor<1x64xf16>
  }

  // Interleaved groups keep separate ids (2 for %A, 3 for %X), numbered in
  // order of first appearance and unique across functions.
  // CHECK-LABEL: func.func @interleaved
  func.func @interleaved(%ctx: !hip.context, %A: tensor<1x256xf16>,
                         %X: tensor<1x256xf16>, %B: tensor<64x8x16xui8>,
                         %S: tensor<64x8xf16>, %E: tensor<1x64xf16>)
      -> (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>,
          tensor<1x64xf16>) {
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 2 : i64, hip.act_quant_leader,
    %0 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 3 : i64, hip.act_quant_leader,
    %1 = hip.matmul_nbits(%ctx) ins(%X, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 2 : i64, zp_elem_size
    %2 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}hip.act_quant_group = 3 : i64, zp_elem_size
    %3 = hip.matmul_nbits(%ctx) ins(%X, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0, %1, %2, %3 : tensor<1x64xf16>, tensor<1x64xf16>,
        tensor<1x64xf16>, tensor<1x64xf16>
  }

  // A sibling inside a conditional region may not run, so it cannot rely on
  // a leader outside it (nor lead one): neither op is tagged.
  // CHECK-LABEL: func.func @other_block
  func.func @other_block(%ctx: !hip.context, %c: i1, %A: tensor<1x256xf16>,
                         %B: tensor<64x8x16xui8>, %S: tensor<64x8xf16>,
                         %E: tensor<1x64xf16>)
      -> (tensor<1x64xf16>, tensor<1x64xf16>) {
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    %1 = scf.if %c -> (tensor<1x64xf16>) {
      // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, zp_elem_size
      %2 = hip.matmul_nbits(%ctx) ins(%A, %B, %S : tensor<1x256xf16>,
          tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
          {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
           accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
      scf.yield %2 : tensor<1x64xf16>
    } else {
      scf.yield %E : tensor<1x64xf16>
    }
    return %0, %1 : tensor<1x64xf16>, tensor<1x64xf16>
  }

  // A pre-mix output leads its group itself, even for a single consumer: the
  // producer gets the group and the consumers' block size, and no consumer is
  // a leader.
  // CHECK-LABEL: func.func @pre_mix_producer
  func.func @pre_mix_producer(%ctx: !hip.context, %X: tensor<1x512xf16>,
                              %G: tensor<1x512xf16>, %Y: tensor<1x256xf16>,
                              %B: tensor<64x8x16xui8>, %S: tensor<64x8xf16>,
                              %E: tensor<1x64xf16>) -> tensor<1x64xf16> {
    // CHECK: hip.hyper_connection_pre_mix({{.*}}hidden = 256 : i64, hip.act_quant_block_size = 32 : i64, hip.act_quant_group = 4 : i64, reduction_scale
    %a = hip.hyper_connection_pre_mix(%ctx) ins(%X, %G : tensor<1x512xf16>,
        tensor<1x512xf16>) outs(%Y : tensor<1x256xf16>)
        {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
         hidden = 256 : i64, reduction_scale = 5.000000e-01 : f32}
        : tensor<1x256xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 4 : i64, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%a, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
  }

  // Only post-mix-norm's normed result (#1) is written as int8; a reader of
  // the residual (#0) is an ordinary input.
  // CHECK-LABEL: func.func @post_mix_norm_producer
  func.func @post_mix_norm_producer(%ctx: !hip.context, %X: tensor<1x512xf16>,
                                    %O: tensor<1x256xf16>, %G: tensor<1x2xf16>,
                                    %W: tensor<512xf16>, %R: tensor<1x512xf16>,
                                    %N: tensor<1x512xf16>,
                                    %B: tensor<64x16x16xui8>,
                                    %S: tensor<64x16xf16>,
                                    %E: tensor<1x64xf16>)
      -> (tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>) {
    // CHECK: hip.hyper_connection_post_mix_norm({{.*}}hidden = 256 : i64, hip.act_quant_block_size = 32 : i64, hip.act_quant_group = 5 : i64}
    %r:2 = hip.hyper_connection_post_mix_norm(%ctx) ins(%X, %O, %G, %W :
        tensor<1x512xf16>, tensor<1x256xf16>, tensor<1x2xf16>, tensor<512xf16>)
        outs(%R, %N : tensor<1x512xf16>, tensor<1x512xf16>)
        {branches = 2 : i64, epsilon = 9.99999997E-7 : f32, flattened = 1 : i64,
         gate_activation = 1 : i64, gate_layout = "branch", hidden = 256 : i64}
        : tensor<1x512xf16>, tensor<1x512xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%r#0, %B, %S : tensor<1x512xf16>,
        tensor<64x16x16xui8>, tensor<64x16xf16>) outs(%E : tensor<1x64xf16>)
        {K = 512 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 5 : i64, zp_elem_size
    %1 = hip.matmul_nbits(%ctx) ins(%r#1, %B, %S : tensor<1x512xf16>,
        tensor<64x16x16xui8>, tensor<64x16xf16>) outs(%E : tensor<1x64xf16>)
        {K = 512 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 5 : i64, zp_elem_size
    %2 = hip.matmul_nbits(%ctx) ins(%r#1, %B, %S : tensor<1x512xf16>,
        tensor<64x16x16xui8>, tensor<64x16xf16>) outs(%E : tensor<1x64xf16>)
        {K = 512 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0, %1, %2 : tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>
  }

  // Consumers that quantize with different block sizes cannot share one
  // producer-written copy; they fall back to an ordinary sibling group.
  // CHECK-LABEL: func.func @producer_mixed_block_size
  func.func @producer_mixed_block_size(%ctx: !hip.context,
                                       %X: tensor<1x512xf16>,
                                       %G: tensor<1x512xf16>,
                                       %Y: tensor<1x256xf16>,
                                       %B: tensor<64x8x16xui8>,
                                       %S: tensor<64x8xf16>,
                                       %B2: tensor<64x4x32xui8>,
                                       %S2: tensor<64x4xf16>,
                                       %E: tensor<1x64xf16>)
      -> (tensor<1x64xf16>, tensor<1x64xf16>) {
    // CHECK: hip.hyper_connection_pre_mix({{.*}}hidden = 256 : i64, reduction_scale
    %a = hip.hyper_connection_pre_mix(%ctx) ins(%X, %G : tensor<1x512xf16>,
        tensor<1x512xf16>) outs(%Y : tensor<1x256xf16>)
        {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
         hidden = 256 : i64, reduction_scale = 5.000000e-01 : f32}
        : tensor<1x256xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 6 : i64, hip.act_quant_leader, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%a, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 64 : i64, hip.act_quant_group = 6 : i64, zp_elem_size
    %1 = hip.matmul_nbits(%ctx) ins(%a, %B2, %S2 : tensor<1x256xf16>,
        tensor<64x4x32xui8>, tensor<64x4xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 64 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0, %1 : tensor<1x64xf16>, tensor<1x64xf16>
  }

  // A mixer writes the int8 copy of its pre-mix output (#0) only.
  // CHECK-LABEL: func.func @mixer_producer
  func.func @mixer_producer(%ctx: !hip.context, %X: tensor<1x512xf16>,
                            %P: tensor<1x66xf16>, %WB: tensor<512x2x16xui8>,
                            %WS: tensor<512x2xf16>, %Y: tensor<1x256xf16>,
                            %GI: tensor<1x2xf16>, %A: tensor<1x64xf16>,
                            %M: tensor<1x512xf16>, %B: tensor<64x8x16xui8>,
                            %S: tensor<64x8xf16>, %E: tensor<1x64xf16>)
      -> tensor<1x64xf16> {
    // CHECK: hip.hyper_connection_mixer({{.*}}hip.act_quant_block_size = 32 : i64, hip.act_quant_group = 7 : i64
    %r:4 = hip.hyper_connection_mixer(%ctx) ins(%X, %P, %WB, %WS :
        tensor<1x512xf16>, tensor<1x66xf16>, tensor<512x2x16xui8>,
        tensor<512x2xf16>)
        outs(%Y, %GI, %A, %M : tensor<1x256xf16>, tensor<1x2xf16>,
             tensor<1x64xf16>, tensor<1x512xf16>)
        {act_scale = 2.500000e-01 : f32, act_width = 64 : i64,
         block_size = 32 : i64, branches = 2 : i64, hidden = 256 : i64,
         reduction_scale = 5.000000e-01 : f32, swish_alpha = 1.000000e+00 : f32}
        : tensor<1x256xf16>, tensor<1x2xf16>, tensor<1x64xf16>,
          tensor<1x512xf16>
    // CHECK: hip.matmul_nbits({{.*}}block_size = 32 : i64, hip.act_quant_group = 7 : i64, zp_elem_size
    %0 = hip.matmul_nbits(%ctx) ins(%r#0, %B, %S : tensor<1x256xf16>,
        tensor<64x8x16xui8>, tensor<64x8xf16>) outs(%E : tensor<1x64xf16>)
        {K = 256 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x64xf16>
    return %0 : tensor<1x64xf16>
  }
}
