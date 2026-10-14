// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// MatMulNBitsGroup merges up to three hip.matmul_nbits ops that read the same
// shared int8 activation into one hip.matmul_nbits_group, and a gate/up pair
// consumed only by SiLU(gate) * up into one group with the silu_mul epilogue.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// q/k/v: three projections of one activation become one group, in program
// order, each original result replaced by the matching group result. The
// later members' inits are defined after the first projection and are moved
// above the group.
// CHECK-LABEL: func.func @qkv
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act
// CHECK:         %[[G:.*]]:3 = hip.matmul_nbits_group(%arg0) ins(%arg1 : tensor<1x?x64xf16>)
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK-SAME:      weights(%arg2, %arg4, %arg6 :
// CHECK-SAME:      scales(%arg3, %arg5, %arg7 :
// CHECK-SAME:      outs(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>)
// CHECK-SAME:      N = [64, 32, 32]
// CHECK-NOT:       epilogue = 1
// CHECK-NOT:     hip.matmul_nbits(
// CHECK:         return %[[G]]#0, %[[G]]#1, %[[G]]#2
func.func @qkv(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
               %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
               %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>,
               %bv: tensor<32x2x16xui8>, %sv: tensor<32x2xf16>)
    -> (tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>) {
  %c1 = arith.constant 1 : index
  %m = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %eq = tensor.empty(%m) : tensor<1x?x64xf16>
  %q = hip.matmul_nbits(%ctx) ins(%x, %bq, %sq :
      tensor<1x?x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%eq : tensor<1x?x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x64xf16>
  %ek = tensor.empty(%m) : tensor<1x?x32xf16>
  %k = hip.matmul_nbits(%ctx) ins(%x, %bk, %sk :
      tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ek : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x32xf16>
  %ev = tensor.empty(%m) : tensor<1x?x32xf16>
  %v = hip.matmul_nbits(%ctx) ins(%x, %bv, %sv :
      tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ev : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x32xf16>
  return %q, %k, %v : tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>
}

// -----

// A later projection whose init is sized from the first projection's result
// cannot move above it, so it is left out; the remaining members still group
// and the one left out keeps reading the shared activation.
// CHECK-LABEL: func.func @member_depends_on_first
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group(
// CHECK-SAME:      weights(%arg2, %arg4 :
// CHECK:         %[[D:.*]] = tensor.dim %[[G]]#0
// CHECK:         %[[E:.*]] = tensor.empty(%[[D]])
// CHECK:         %[[V:.*]] = hip.matmul_nbits(%arg0) ins(%arg1, %arg6, %arg7 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK-SAME:      outs(%[[E]] :
// CHECK:         return %[[G]]#0, %[[G]]#1, %[[V]]
func.func @member_depends_on_first(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
                                   %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
                                   %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>,
                                   %bv: tensor<32x2x16xui8>, %sv: tensor<32x2xf16>)
    -> (tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>) {
  %c1 = arith.constant 1 : index
  %m = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %eq = tensor.empty(%m) : tensor<1x?x64xf16>
  %q = hip.matmul_nbits(%ctx) ins(%x, %bq, %sq :
      tensor<1x?x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%eq : tensor<1x?x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x64xf16>
  %ek = tensor.empty(%m) : tensor<1x?x32xf16>
  %k = hip.matmul_nbits(%ctx) ins(%x, %bk, %sk :
      tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ek : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x32xf16>
  %mq = tensor.dim %q, %c1 : tensor<1x?x64xf16>
  %ev = tensor.empty(%mq) : tensor<1x?x32xf16>
  %v = hip.matmul_nbits(%ctx) ins(%x, %bv, %sv :
      tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ev : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x32xf16>
  return %q, %k, %v : tensor<1x?x64xf16>, tensor<1x?x32xf16>, tensor<1x?x32xf16>
}

// -----

// Four readers: the first three form the group (the kernel's limit) and the
// fourth stays a standalone projection of the shared activation.
// CHECK-LABEL: func.func @four_readers
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act
// CHECK:         hip.matmul_nbits_group(
// CHECK-SAME:      weights(%arg2, %arg3, %arg4 :
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg5, %arg6 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK-NOT:     hip.matmul_nbits_group
func.func @four_readers(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                        %b0: tensor<32x2x16xui8>, %b1: tensor<32x2x16xui8>,
                        %b2: tensor<32x2x16xui8>, %b3: tensor<32x2x16xui8>,
                        %s: tensor<32x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>,
        tensor<1x1x32xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %b0, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %b1, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x32xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x, %b2, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e2 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  %e3 = tensor.empty() : tensor<1x1x32xf16>
  %y3 = hip.matmul_nbits(%ctx) ins(%x, %b3, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e3 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  return %y0, %y1, %y2, %y3 : tensor<1x1x32xf16>, tensor<1x1x32xf16>,
      tensor<1x1x32xf16>, tensor<1x1x32xf16>
}

// -----

// gate/up feeding SiLU(gate) * up: the sigmoid and both muls fold into the
// group's epilogue. The group writes into the gate projection's init, and the
// up projection's init stays on the op as up_tmp, a plain transient.
// CHECK-LABEL: func.func @gate_up_silu
// CHECK:         %[[EG:.*]] = tensor.empty() : tensor<1x1x128xf16>
// CHECK:         %[[EU:.*]] = tensor.empty() : tensor<1x1x128xf16>
// CHECK:         %[[Y:.*]]:2 = hip.matmul_nbits_group(%arg0) ins(%arg1 : tensor<1x1x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg4 :
// CHECK-SAME:      scales(%arg3, %arg5 :
// CHECK-SAME:      outs(%[[EG]] : tensor<1x1x128xf16>)
// CHECK-SAME:      up_tmp(%[[EU]] : tensor<1x1x128xf16>)
// CHECK-SAME:      N = [128, 128]
// CHECK-SAME:      epilogue = 1
// CHECK-NOT:     hip.sigmoid
// CHECK-NOT:     hip.mul
// CHECK:         return %[[Y]]#0
func.func @gate_up_silu(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                        %bg: tensor<128x2x16xui8>, %sg: tensor<128x2xf16>,
                        %bu: tensor<128x2x16xui8>, %su: tensor<128x2xf16>)
    -> tensor<1x1x128xf16> {
  %eg = tensor.empty() : tensor<1x1x128xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %bg, %sg :
      tensor<1x1x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eg : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x128xf16>
  %eu = tensor.empty() : tensor<1x1x128xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %bu, %su :
      tensor<1x1x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eu : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x128xf16>
  %es = tensor.empty() : tensor<1x1x128xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x1x128xf16>)
      outs(%es : tensor<1x1x128xf16>) : tensor<1x1x128xf16>
  %em = tensor.empty() : tensor<1x1x128xf16>
  %gs = hip.mul(%ctx) ins(%g, %s : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      outs(%em : tensor<1x1x128xf16>) -> tensor<1x1x128xf16>
  %ey = tensor.empty() : tensor<1x1x128xf16>
  %y = hip.mul(%ctx) ins(%gs, %u : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      outs(%ey : tensor<1x1x128xf16>) -> tensor<1x1x128xf16>
  return %y : tensor<1x1x128xf16>
}

// -----

// The conversions size each init from `tensor.dim` of its producer, so the
// gate and the intermediate products also feed shape queries. Those are not
// consumers of the values; they are re-pointed at the fused result, which has
// the same shape. Here up comes first in program order, and the group still
// lists gate as segment 0.
// CHECK-LABEL: func.func @gate_up_silu_dynamic
// CHECK:         %[[Y:.*]]:2 = hip.matmul_nbits_group
// CHECK-SAME:      weights(%arg2, %arg4 :
// CHECK-SAME:      epilogue = 1
// CHECK-NOT:     hip.sigmoid
// CHECK-NOT:     hip.mul
// CHECK:         return %[[Y]]#0
func.func @gate_up_silu_dynamic(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
                                %bg: tensor<128x2x16xui8>, %sg: tensor<128x2xf16>,
                                %bu: tensor<128x2x16xui8>, %su: tensor<128x2xf16>)
    -> tensor<1x?x128xf16> {
  %c1 = arith.constant 1 : index
  %m = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %eu = tensor.empty(%m) : tensor<1x?x128xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %bu, %su :
      tensor<1x?x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eu : tensor<1x?x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x128xf16>
  %eg = tensor.empty(%m) : tensor<1x?x128xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %bg, %sg :
      tensor<1x?x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eg : tensor<1x?x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x128xf16>
  %d0 = tensor.dim %g, %c1 : tensor<1x?x128xf16>
  %es = tensor.empty(%d0) : tensor<1x?x128xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x?x128xf16>)
      outs(%es : tensor<1x?x128xf16>) : tensor<1x?x128xf16>
  %d1 = tensor.dim %g, %c1 : tensor<1x?x128xf16>
  %em = tensor.empty(%d1) : tensor<1x?x128xf16>
  %gs = hip.mul(%ctx) ins(%g, %s : tensor<1x?x128xf16>, tensor<1x?x128xf16>)
      outs(%em : tensor<1x?x128xf16>) -> tensor<1x?x128xf16>
  %d2 = tensor.dim %gs, %c1 : tensor<1x?x128xf16>
  %ey = tensor.empty(%d2) : tensor<1x?x128xf16>
  %y = hip.mul(%ctx) ins(%gs, %u : tensor<1x?x128xf16>, tensor<1x?x128xf16>)
      outs(%ey : tensor<1x?x128xf16>) -> tensor<1x?x128xf16>
  return %y : tensor<1x?x128xf16>
}

// -----

// The gate projection has a second consumer, so it has to reach memory: the
// pair is still grouped, without the epilogue, and the SiLU chain stays.
// CHECK-LABEL: func.func @gate_escapes
// CHECK:         hip.matmul_nbits_group
// CHECK-NOT:       up_tmp
// CHECK-NOT:       epilogue = 1
// CHECK:         hip.sigmoid
// CHECK:         hip.mul
// CHECK:         hip.mul
func.func @gate_escapes(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                        %bg: tensor<128x2x16xui8>, %sg: tensor<128x2xf16>,
                        %bu: tensor<128x2x16xui8>, %su: tensor<128x2xf16>)
    -> (tensor<1x1x128xf16>, tensor<1x1x128xf16>) {
  %eg = tensor.empty() : tensor<1x1x128xf16>
  %g = hip.matmul_nbits(%ctx) ins(%x, %bg, %sg :
      tensor<1x1x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eg : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x128xf16>
  %eu = tensor.empty() : tensor<1x1x128xf16>
  %u = hip.matmul_nbits(%ctx) ins(%x, %bu, %su :
      tensor<1x1x64xf16>, tensor<128x2x16xui8>, tensor<128x2xf16>)
      outs(%eu : tensor<1x1x128xf16>)
      {K = 64 : i64, N = 128 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x128xf16>
  %es = tensor.empty() : tensor<1x1x128xf16>
  %s = hip.sigmoid(%ctx) ins(%g : tensor<1x1x128xf16>)
      outs(%es : tensor<1x1x128xf16>) : tensor<1x1x128xf16>
  %em = tensor.empty() : tensor<1x1x128xf16>
  %gs = hip.mul(%ctx) ins(%g, %s : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      outs(%em : tensor<1x1x128xf16>) -> tensor<1x1x128xf16>
  %ey = tensor.empty() : tensor<1x1x128xf16>
  %y = hip.mul(%ctx) ins(%gs, %u : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      outs(%ey : tensor<1x1x128xf16>) -> tensor<1x1x128xf16>
  return %y, %g : tensor<1x1x128xf16>, tensor<1x1x128xf16>
}

// -----

// A group fed by hip.skip_rms_norm reads the norm's quantized copy of the
// row (quant_outs) and the norm's existing results keep their positions.
// CHECK-LABEL: func.func @norm_feeds_group
// CHECK:         %[[N:.*]]:4 = hip.skip_rms_norm
// CHECK-SAME:      quant_outs(%{{.*}}, %{{.*}} : tensor<1x1x64xi8>, tensor<1x1x2xf32>)
// CHECK-NOT:     hip.quantize_act
// CHECK:         hip.matmul_nbits_group(%arg0) ins(%[[N]]#0 : tensor<1x1x64xf16>)
// CHECK-SAME:      quantized(%[[N]]#2, %[[N]]#3 : tensor<1x1x64xi8>, tensor<1x1x2xf32>)
// CHECK:         return {{.*}}%[[N]]#1
func.func @norm_feeds_group(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                            %r: tensor<1x1x64xf16>, %gamma: tensor<64xf16>,
                            %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
                            %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x64xf16>) {
  %ey = tensor.empty() : tensor<1x1x64xf16>
  %es = tensor.empty() : tensor<1x1x64xf16>
  %y, %sum = hip.skip_rms_norm(%ctx)
      ins(%x, %r, %gamma : tensor<1x1x64xf16>, tensor<1x1x64xf16>, tensor<64xf16>)
      outs(%ey, %es : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      {epsilon = 1.0e-05 : f32} : tensor<1x1x64xf16>, tensor<1x1x64xf16>
  %eq = tensor.empty() : tensor<1x1x64xf16>
  %q = hip.matmul_nbits(%ctx) ins(%y, %bq, %sq :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%eq : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %ek = tensor.empty() : tensor<1x1x32xf16>
  %k = hip.matmul_nbits(%ctx) ins(%y, %bk, %sk :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ek : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  return %q, %k, %sum : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x64xf16>
}

// -----

// Asymmetric weights with packed uint8 zero points (AWQ) are grouped too; each
// member's zero points follow its weights into the group, in member order.
// CHECK-LABEL: func.func @zero_points_grouped
// CHECK:         hip.matmul_nbits_group(%arg0) ins(%arg1 : tensor<1x1x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg5 :
// CHECK-SAME:      scales(%arg3, %arg6 :
// CHECK-SAME:      zero_points(%arg4, %arg7 : tensor<32x1xui8>, tensor<16x1xui8>)
// CHECK-NOT:     hip.matmul_nbits(
func.func @zero_points_grouped(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                               %b0: tensor<32x2x16xui8>, %s0: tensor<32x2xf16>,
                               %z0: tensor<32x1xui8>,
                               %b1: tensor<16x2x16xui8>, %s1: tensor<16x2xf16>,
                               %z1: tensor<16x1xui8>)
    -> (tensor<1x1x32xf16>, tensor<1x1x16xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      zero_points(%z0 : tensor<32x1xui8>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x16xf16>
  %c = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      zero_points(%z1 : tensor<16x1xui8>)
      outs(%e1 : tensor<1x1x16xf16>)
      {K = 64 : i64, N = 16 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x1x16xf16>
  return %a, %c : tensor<1x1x32xf16>, tensor<1x1x16xf16>
}

// -----

// A symmetric and an asymmetric projection of the same row share the int8
// activation but are not grouped: the group's zero points are all or nothing.
// CHECK-LABEL: func.func @mixed_zero_points_not_grouped
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK-COUNT-2: hip.matmul_nbits(
func.func @mixed_zero_points_not_grouped(%ctx: !hip.context,
                                         %x: tensor<1x1x64xf16>,
                                         %b: tensor<32x2x16xui8>,
                                         %s: tensor<32x2xf16>,
                                         %zp: tensor<32x1xui8>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      zero_points(%zp : tensor<32x1xui8>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %c = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  return %a, %c : tensor<1x1x32xf16>, tensor<1x1x32xf16>
}
