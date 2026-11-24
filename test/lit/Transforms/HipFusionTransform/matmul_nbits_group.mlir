// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// MatMulNBitsGroup folds hip.matmul_nbits ops that read the same activation
// into one hip.matmul_nbits_group, and a gate/up pair consumed only by
// SiLU(gate) * up into one group with the silu_mul epilogue.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// q/k/v: three projections of one activation become one group, in program
// order, each original result replaced by the matching group result. The
// later members' inits are defined after the first projection and have to be
// moved above the group.
// CHECK-LABEL: func.func @qkv
// CHECK:         %[[G:.*]]:3 = hip.matmul_nbits_group(%{{.*}}) ins(%{{.*}} : tensor<1x?x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg4, %arg6 :
// CHECK-SAME:      outs(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} :
// CHECK-SAME:      N = [64, 32, 32]
// CHECK-NOT:       epilogue
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

// Four projections of one activation (Qwen3.8-Flash-Next's N = 1, 512, 640,
// 640 attention inputs, scaled down) form a single group.
// CHECK-LABEL: func.func @four_way
// CHECK:         %[[G:.*]]:4 = hip.matmul_nbits_group(
// CHECK-SAME:      weights(%arg2, %arg4, %arg6, %arg8 :
// CHECK-SAME:      N = [8, 32, 64, 64]
// CHECK-NOT:     hip.matmul_nbits(
// CHECK:         return %[[G]]#0, %[[G]]#1, %[[G]]#2, %[[G]]#3
func.func @four_way(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                    %b0: tensor<8x2x16xui8>, %s0: tensor<8x2xf16>,
                    %b1: tensor<32x2x16xui8>, %s1: tensor<32x2xf16>,
                    %b2: tensor<64x2x16xui8>, %s2: tensor<64x2xf16>,
                    %b3: tensor<64x2x16xui8>, %s3: tensor<64x2xf16>)
    -> (tensor<1x1x8xf16>, tensor<1x1x32xf16>, tensor<1x1x64xf16>,
        tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x8xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x64xf16>, tensor<8x2x16xui8>, tensor<8x2xf16>)
      outs(%e0 : tensor<1x1x8xf16>)
      {K = 64 : i64, N = 8 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x8xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x, %b2, %s2 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%e2 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %e3 = tensor.empty() : tensor<1x1x64xf16>
  %y3 = hip.matmul_nbits(%ctx) ins(%x, %b3, %s3 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%e3 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  return %y0, %y1, %y2, %y3 : tensor<1x1x8xf16>, tensor<1x1x32xf16>,
      tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// A later projection whose init is sized from the first projection's result
// cannot move above it, so it is left out; the remaining members still group.
// CHECK-LABEL: func.func @member_depends_on_first
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group(
// CHECK-SAME:      weights(%arg2, %arg4 :
// CHECK:         %[[D:.*]] = tensor.dim %[[G]]#0
// CHECK:         %[[E:.*]] = tensor.empty(%[[D]])
// CHECK:         %[[V:.*]] = hip.matmul_nbits(%{{.*}}) ins(%{{.*}}, %arg6, %arg7 :
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

// gate/up feeding SiLU(gate) * up: the sigmoid and both muls fold into the
// group's epilogue and the group writes into the gate projection's init.
// CHECK-LABEL: func.func @gate_up_silu
// CHECK:         %[[EG:.*]] = tensor.empty
// CHECK:         %[[Y:.*]] = hip.matmul_nbits_group(%{{.*}}) ins(%{{.*}} : tensor<1x1x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg4 :
// CHECK-SAME:      scales(%arg3, %arg5 :
// CHECK-SAME:      outs(%[[EG]] : tensor<1x1x128xf16>)
// CHECK-SAME:      epilogue = 1
// CHECK-NOT:     hip.sigmoid
// CHECK-NOT:     hip.mul
// CHECK:         return %[[Y]]
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
// the same shape.
// CHECK-LABEL: func.func @gate_up_silu_dynamic
// CHECK:         %[[Y:.*]] = hip.matmul_nbits_group
// CHECK-SAME:      epilogue = 1
// CHECK-NOT:     hip.sigmoid
// CHECK-NOT:     hip.mul
// CHECK:         return %[[Y]]
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
// pair is still grouped, but with the concat epilogue, and the SiLU chain
// stays (as the hip.swiglu SwigluFusion makes of it).
// CHECK-LABEL: func.func @gate_escapes
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group
// CHECK-NOT:       epilogue
// CHECK:         hip.swiglu(%{{.*}}) ins(%[[G]]#0, %[[G]]#1 :
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
// row: the norm grows quant_outs (int8 row, one f32 scale per 32 elements)
// and the group takes them as quantized(...). Existing norm results keep
// their positions.
// CHECK-LABEL: func.func @norm_feeds_group
// CHECK:         %[[N:.*]]:4 = hip.skip_rms_norm
// CHECK-SAME:      quant_outs(%{{.*}}, %{{.*}} : tensor<1x1x64xi8>, tensor<1x1x2xf32>)
// CHECK-SAME:      quant_block_size = 32
// CHECK:         hip.matmul_nbits_group(%{{.*}}) ins(%[[N]]#0 : tensor<1x1x64xf16>)
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

// hip.hyper_connection_pre_mix feeding a group writes the quantized copy of
// its output the same way.
// CHECK-LABEL: func.func @pre_mix_feeds_group
// CHECK:         %[[P:.*]]:3 = hip.hyper_connection_pre_mix
// CHECK-SAME:      quant_outs(%{{.*}}, %{{.*}} : tensor<1x1x32xi8>, tensor<1x1x1xf32>)
// CHECK-SAME:      quant_block_size = 32
// CHECK:         hip.matmul_nbits_group(%{{.*}}) ins(%[[P]]#0 : tensor<1x1x32xf16>)
// CHECK-SAME:      quantized(%[[P]]#1, %[[P]]#2 : tensor<1x1x32xi8>, tensor<1x1x1xf32>)
func.func @pre_mix_feeds_group(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                               %g: tensor<1x1x64xf16>,
                               %bq: tensor<64x1x16xui8>, %sq: tensor<64x1xf16>,
                               %bk: tensor<32x1x16xui8>, %sk: tensor<32x1xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x32xf16>) {
  %ey = tensor.empty() : tensor<1x1x32xf16>
  %y = hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      outs(%ey : tensor<1x1x32xf16>)
      {branches = 2 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 32 : i64, reduction_scale = 5.000000e-01 : f32}
      : tensor<1x1x32xf16>
  %eq = tensor.empty() : tensor<1x1x64xf16>
  %q = hip.matmul_nbits(%ctx) ins(%y, %bq, %sq :
      tensor<1x1x32xf16>, tensor<64x1x16xui8>, tensor<64x1xf16>)
      outs(%eq : tensor<1x1x64xf16>)
      {K = 32 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %ek = tensor.empty() : tensor<1x1x32xf16>
  %k = hip.matmul_nbits(%ctx) ins(%y, %bk, %sk :
      tensor<1x1x32xf16>, tensor<32x1x16xui8>, tensor<32x1xf16>)
      outs(%ek : tensor<1x1x32xf16>)
      {K = 32 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  return %q, %k : tensor<1x1x64xf16>, tensor<1x1x32xf16>
}

// -----

// The flattened normed output of hip.hyper_connection_post_mix_norm (result
// 1, one row of branches * hidden) feeding a group is quantized in-kernel;
// the residual keeps result 0.
// CHECK-LABEL: func.func @post_mix_norm_feeds_group
// CHECK:         %[[P:.*]]:4 = hip.hyper_connection_post_mix_norm
// CHECK-SAME:      quant_outs(%{{.*}}, %{{.*}} : tensor<1x1x64xi8>, tensor<1x1x2xf32>)
// CHECK-SAME:      quant_block_size = 32
// CHECK:         hip.matmul_nbits_group(%{{.*}}) ins(%[[P]]#1 : tensor<1x1x64xf16>)
// CHECK-SAME:      quantized(%[[P]]#2, %[[P]]#3 : tensor<1x1x64xi8>, tensor<1x1x2xf32>)
// CHECK:         return {{.*}}%[[P]]#0
func.func @post_mix_norm_feeds_group(
    %ctx: !hip.context, %x: tensor<1x1x64xf16>, %b: tensor<1x1x32xf16>,
    %g: tensor<1x1x2xf16>, %w: tensor<64xf16>,
    %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
    %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x64xf16>) {
  %er = tensor.empty() : tensor<1x1x64xf16>
  %en = tensor.empty() : tensor<1x1x64xf16>
  %r, %n = hip.hyper_connection_post_mix_norm(%ctx)
      ins(%x, %b, %g, %w : tensor<1x1x64xf16>, tensor<1x1x32xf16>,
          tensor<1x1x2xf16>, tensor<64xf16>)
      outs(%er, %en : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
      {branches = 2 : i64, epsilon = 1.0e-06 : f32, flattened = 1 : i64,
       gate_activation = 1 : i64, gate_layout = "branch", hidden = 32 : i64}
      : tensor<1x1x64xf16>, tensor<1x1x64xf16>
  %eq = tensor.empty() : tensor<1x1x64xf16>
  %q = hip.matmul_nbits(%ctx) ins(%n, %bq, %sq :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%eq : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %ek = tensor.empty() : tensor<1x1x32xf16>
  %k = hip.matmul_nbits(%ctx) ins(%n, %bk, %sk :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ek : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x32xf16>
  return %q, %k, %r : tensor<1x1x64xf16>, tensor<1x1x32xf16>, tensor<1x1x64xf16>
}

// -----

// Asymmetric weights with packed uint8 zero points (AWQ) are grouped too; each
// member's zero points follow its weights into the group, in member order.
// CHECK-LABEL: func.func @zero_points_grouped
// CHECK:         hip.matmul_nbits_group(%{{.*}}) ins(%{{.*}} : tensor<1x1x64xf16>)
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

// A symmetric and an asymmetric projection of the same row are not grouped:
// the group's zero points are all or nothing.
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

// -----

// fp16 zero points (zp_elem_size = 2) are not grouped: the grouped kernel
// reads the uint8 unpack of packed zero points.
// CHECK-LABEL: func.func @fp16_zero_points_not_grouped
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK-COUNT-2: hip.matmul_nbits(
func.func @fp16_zero_points_not_grouped(%ctx: !hip.context,
                                        %x: tensor<1x1x64xf16>,
                                        %b: tensor<32x2x16xui8>,
                                        %s: tensor<32x2xf16>,
                                        %zp: tensor<32x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>) {
  %e0 = tensor.empty() : tensor<1x1x32xf16>
  %a = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      zero_points(%zp : tensor<32x2xf16>)
      outs(%e0 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  %e1 = tensor.empty() : tensor<1x1x32xf16>
  %c = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      zero_points(%zp : tensor<32x2xf16>)
      outs(%e1 : tensor<1x1x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x32xf16>
  return %a, %c : tensor<1x1x32xf16>, tensor<1x1x32xf16>
}

// -----

// A projection read only through last-axis slices that tile its output (a
// fused projection split back into its parts) becomes a group with one member
// per slice, each reading its rows of the same weight. No strided slice of the
// full output is left for bufferization to copy out.
// CHECK-LABEL: func.func @sliced_output
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group(%{{.*}}) ins(%arg1 : tensor<?x?x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg2 :
// CHECK-SAME:      scales(%arg3, %arg3 :
// CHECK-SAME:      outs(%{{.*}}, %{{.*}} : tensor<?x?x8xf16>, tensor<?x?x4xf16>)
// CHECK-SAME:      N = [8, 4]
// CHECK-SAME:      row_offsets = array<i64: 0, 8>
// CHECK-NOT:     tensor.extract_slice
// CHECK-NOT:     hip.matmul_nbits(
// CHECK:         hip.swish(%{{.*}}) ins(%[[G]]#0 :
// CHECK:         return %{{.*}}, %[[G]]#1
func.func @sliced_output(%ctx: !hip.context, %x: tensor<?x?x64xf16>,
                         %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>)
    -> (tensor<?x?x8xf16>, tensor<?x?x4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %x, %c0 : tensor<?x?x64xf16>
  %d1 = tensor.dim %x, %c1 : tensor<?x?x64xf16>
  %e = tensor.empty(%d0, %d1) : tensor<?x?x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<?x?x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<?x?x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<?x?x12xf16>
  %y0 = tensor.dim %y, %c0 : tensor<?x?x12xf16>
  %y1 = tensor.dim %y, %c1 : tensor<?x?x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [%y0, %y1, 8] [1, 1, 1]
      : tensor<?x?x12xf16> to tensor<?x?x8xf16>
  %g = tensor.extract_slice %y[0, 0, 8] [%y0, %y1, 4] [1, 1, 1]
      : tensor<?x?x12xf16> to tensor<?x?x4xf16>
  %ea = tensor.empty(%y0, %y1) : tensor<?x?x8xf16>
  %sw = hip.swish(%ctx) ins(%a : tensor<?x?x8xf16>) outs(%ea : tensor<?x?x8xf16>)
      : tensor<?x?x8xf16>
  return %sw, %g : tensor<?x?x8xf16>, tensor<?x?x4xf16>
}

// -----

// The same inside a group: the sliced member becomes two members at row
// offsets 0 and 32 of its weight, the other member keeps its whole weight.
// CHECK-LABEL: func.func @sliced_group_member
// CHECK:         %[[G:.*]]:3 = hip.matmul_nbits_group(
// CHECK-SAME:      weights(%arg2, %arg2, %arg4 :
// CHECK-SAME:      N = [32, 32, 16]
// CHECK-SAME:      row_offsets = array<i64: 0, 32, 0>
// CHECK-NOT:     tensor.extract_slice
// CHECK:         return %[[G]]#0, %[[G]]#1, %[[G]]#2
func.func @sliced_group_member(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                               %b0: tensor<64x2x16xui8>, %s0: tensor<64x2xf16>,
                               %b1: tensor<16x2x16xui8>, %s1: tensor<16x2xf16>)
    -> (tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x16xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x16xf16>
  %z = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x64xf16>, tensor<16x2x16xui8>, tensor<16x2xf16>)
      outs(%e1 : tensor<1x1x16xf16>)
      {K = 64 : i64, N = 16 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x16xf16>
  %lo = tensor.extract_slice %y[0, 0, 0] [1, 1, 32] [1, 1, 1]
      : tensor<1x1x64xf16> to tensor<1x1x32xf16>
  %hi = tensor.extract_slice %y[0, 0, 32] [1, 1, 32] [1, 1, 1]
      : tensor<1x1x64xf16> to tensor<1x1x32xf16>
  return %lo, %hi, %z : tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x16xf16>
}

// -----

// Slices that leave columns unread, or an output read whole as well, keep the
// projection as it is.
// CHECK-LABEL: func.func @slices_not_split
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK-COUNT-2: hip.matmul_nbits(
func.func @slices_not_split(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                            %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>,
                            %x2: tensor<1x1x64xf16>)
    -> (tensor<1x1x8xf16>, tensor<1x1x8xf16>, tensor<1x1x12xf16>) {
  %e = tensor.empty() : tensor<1x1x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<1x1x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [1, 1, 8] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x8xf16>
  %e2 = tensor.empty() : tensor<1x1x12xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x2, %b, %s :
      tensor<1x1x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e2 : tensor<1x1x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x12xf16>
  %a2 = tensor.extract_slice %y2[0, 0, 0] [1, 1, 8] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x8xf16>
  return %a, %a2, %y2 : tensor<1x1x8xf16>, tensor<1x1x8xf16>, tensor<1x1x12xf16>
}

// -----

// An fp32 residual with fp16 skip, gamma and output takes the norm's mixed
// entry, which has no quantized form: the group forms but quantizes itself.
// CHECK-LABEL: func.func @mixed_norm_not_quantized
// CHECK:         hip.skip_rms_norm
// CHECK-NOT:     quant_outs
// CHECK:         hip.matmul_nbits_group
// CHECK-NOT:     quantized(
func.func @mixed_norm_not_quantized(%ctx: !hip.context, %x: tensor<1x1x64xf32>,
                                    %r: tensor<1x1x64xf16>, %gamma: tensor<64xf16>,
                                    %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
                                    %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x32xf16>) {
  %ey = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.skip_rms_norm(%ctx)
      ins(%x, %r, %gamma : tensor<1x1x64xf32>, tensor<1x1x64xf16>, tensor<64xf16>)
      outs(%ey : tensor<1x1x64xf16>)
      {epsilon = 1.0e-05 : f32} : tensor<1x1x64xf16>
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
  return %q, %k : tensor<1x1x64xf16>, tensor<1x1x32xf16>
}

// -----

// Overlapping slices do not tile the output, so it is kept whole.
// CHECK-LABEL: func.func @overlapping_slices_not_split
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK:         hip.matmul_nbits(
// CHECK-COUNT-2: tensor.extract_slice
func.func @overlapping_slices_not_split(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                                        %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>)
    -> (tensor<1x1x8xf16>, tensor<1x1x8xf16>) {
  %e = tensor.empty() : tensor<1x1x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<1x1x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [1, 1, 8] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x8xf16>
  %c = tensor.extract_slice %y[0, 0, 4] [1, 1, 8] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x8xf16>
  return %a, %c : tensor<1x1x8xf16>, tensor<1x1x8xf16>
}

// -----

// A one-column slice would make an N = 1 member, so the output is kept whole.
// CHECK-LABEL: func.func @width_one_slice_not_split
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK:         hip.matmul_nbits(
// CHECK-COUNT-2: tensor.extract_slice
func.func @width_one_slice_not_split(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                                     %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>)
    -> (tensor<1x1x11xf16>, tensor<1x1x1xf16>) {
  %e = tensor.empty() : tensor<1x1x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<1x1x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [1, 1, 11] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x11xf16>
  %c = tensor.extract_slice %y[0, 0, 11] [1, 1, 1] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x1xf16>
  return %a, %c : tensor<1x1x11xf16>, tensor<1x1x1xf16>
}

// -----

// The clamped SwiGLU computes something other than the silu_mul epilogue, so
// the pair is grouped with the concat epilogue and the swiglu stays.
// CHECK-LABEL: func.func @clamped_swiglu_not_fused
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group
// CHECK-NOT:       epilogue
// CHECK:         hip.swiglu(%{{.*}}) ins(%[[G]]#0, %[[G]]#1 :
// CHECK-SAME:      swiglu_limit = 7.000000e+00
func.func @clamped_swiglu_not_fused(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
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
  %ey = tensor.empty() : tensor<1x1x128xf16>
  %y = hip.swiglu(%ctx) ins(%g, %u : tensor<1x1x128xf16>, tensor<1x1x128xf16>)
      outs(%ey : tensor<1x1x128xf16>)
      {activation_alpha = 1.702 : f32, activation_beta = 1.0 : f32,
       swiglu_limit = 7.0 : f32} : tensor<1x1x128xf16>
  return %y : tensor<1x1x128xf16>
}

// -----

// Five slices would need five members; a group holds four.
// CHECK-LABEL: func.func @five_slices_not_split
// CHECK-NOT:     hip.matmul_nbits_group
// CHECK:         hip.matmul_nbits(
// CHECK-COUNT-5: tensor.extract_slice
func.func @five_slices_not_split(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                                 %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>)
    -> (tensor<1x1x2xf16>, tensor<1x1x2xf16>, tensor<1x1x2xf16>,
        tensor<1x1x2xf16>, tensor<1x1x4xf16>) {
  %e = tensor.empty() : tensor<1x1x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<1x1x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x2xf16>
  %c = tensor.extract_slice %y[0, 0, 2] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x2xf16>
  %d = tensor.extract_slice %y[0, 0, 4] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x2xf16>
  %f = tensor.extract_slice %y[0, 0, 6] [1, 1, 2] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x2xf16>
  %g = tensor.extract_slice %y[0, 0, 8] [1, 1, 4] [1, 1, 1]
      : tensor<1x1x12xf16> to tensor<1x1x4xf16>
  return %a, %c, %d, %f, %g : tensor<1x1x2xf16>, tensor<1x1x2xf16>,
      tensor<1x1x2xf16>, tensor<1x1x2xf16>, tensor<1x1x4xf16>
}

// -----

// A sibling projection that cannot join a group with the sliced one (its init
// is sized from the sliced output) does not stop the split.
// CHECK-LABEL: func.func @split_beside_ungroupable_sibling
// CHECK:         %[[G:.*]]:2 = hip.matmul_nbits_group(%{{.*}}) ins(%arg1 : tensor<1x?x64xf16>)
// CHECK-SAME:      weights(%arg2, %arg2 :
// CHECK-SAME:      N = [8, 4]
// CHECK-SAME:      row_offsets = array<i64: 0, 8>
// CHECK:         %[[V:.*]] = hip.matmul_nbits(%{{.*}}) ins(%arg1, %arg4, %arg5 :
// CHECK:         return %[[G]]#0, %[[G]]#1, %[[V]]
func.func @split_beside_ungroupable_sibling(
    %ctx: !hip.context, %x: tensor<1x?x64xf16>,
    %b: tensor<12x2x16xui8>, %s: tensor<12x2xf16>,
    %bv: tensor<32x2x16xui8>, %sv: tensor<32x2xf16>)
    -> (tensor<1x?x8xf16>, tensor<1x?x4xf16>, tensor<1x?x32xf16>) {
  %c1 = arith.constant 1 : index
  %m = tensor.dim %x, %c1 : tensor<1x?x64xf16>
  %e = tensor.empty(%m) : tensor<1x?x12xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x?x64xf16>, tensor<12x2x16xui8>, tensor<12x2xf16>)
      outs(%e : tensor<1x?x12xf16>)
      {K = 64 : i64, N = 12 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x12xf16>
  %my = tensor.dim %y, %c1 : tensor<1x?x12xf16>
  %a = tensor.extract_slice %y[0, 0, 0] [1, %my, 8] [1, 1, 1]
      : tensor<1x?x12xf16> to tensor<1x?x8xf16>
  %g = tensor.extract_slice %y[0, 0, 8] [1, %my, 4] [1, 1, 1]
      : tensor<1x?x12xf16> to tensor<1x?x4xf16>
  %ev = tensor.empty(%my) : tensor<1x?x32xf16>
  %v = hip.matmul_nbits(%ctx) ins(%x, %bv, %sv :
      tensor<1x?x64xf16>, tensor<32x2x16xui8>, tensor<32x2xf16>)
      outs(%ev : tensor<1x?x32xf16>)
      {K = 64 : i64, N = 32 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x?x32xf16>
  return %a, %g, %v : tensor<1x?x8xf16>, tensor<1x?x4xf16>, tensor<1x?x32xf16>
}
