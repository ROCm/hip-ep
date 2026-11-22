// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.matmul_nbits_group lowers to one call of wrap_matmul_nbits_group with a
// fixed four-segment signature: segments the op does not have are passed as
// null pointers with N = 0, and so are absent zero points and an absent
// quantized(...) operand pair.

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @group_qkv(%ctx: !hip.context,
                       %A: memref<1x1x5120xf16, 1>,
                       %Bq: memref<5120x160x16xui8, 1>,
                       %Bk: memref<1024x160x16xui8, 1>,
                       %Bv: memref<1024x160x16xui8, 1>,
                       %Sq: memref<5120x160xf16, 1>,
                       %Sk: memref<1024x160xf16, 1>,
                       %Sv: memref<1024x160xf16, 1>,
                       %q: memref<1x1x5120xf16, 1>,
                       %k: memref<1x1x1024xf16, 1>,
                       %v: memref<1x1x1024xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x5120xf16, 1>)
        weights(%Bq, %Bk, %Bv : memref<5120x160x16xui8, 1>,
                memref<1024x160x16xui8, 1>, memref<1024x160x16xui8, 1>)
        scales(%Sq, %Sk, %Sv : memref<5120x160xf16, 1>,
               memref<1024x160xf16, 1>, memref<1024x160xf16, 1>)
        outs(%q, %k, %v : memref<1x1x5120xf16, 1>, memref<1x1x1024xf16, 1>,
             memref<1x1x1024xf16, 1>)
        {K = 5120 : i64, N = [5120, 1024, 1024], bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @group_qkv
  // CHECK-DAG:   %[[N0:.*]] = llvm.mlir.constant(5120 : i64) : i64
  // CHECK-DAG:   %[[N1:.*]] = llvm.mlir.constant(1024 : i64) : i64
  // CHECK-DAG:   %[[SEG:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK:       llvm.call @wrap_matmul_nbits_group({{.*}}) :
  // CHECK-SAME:  (!llvm.ptr, i32, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr,
  // CHECK-SAME:  !llvm.ptr,
  // CHECK-SAME:  i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
  // CHECK-SAME:  !llvm.ptr, !llvm.ptr) -> i32

  func.func @group_silu(%ctx: !hip.context,
                        %A: memref<1x?x5120xf16, 1>,
                        %Bg: memref<13824x160x16xui8, 1>,
                        %Bu: memref<13824x160x16xui8, 1>,
                        %Sg: memref<13824x160xf16, 1>,
                        %Su: memref<13824x160xf16, 1>,
                        %y: memref<1x?x13824xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x?x5120xf16, 1>)
        weights(%Bg, %Bu : memref<13824x160x16xui8, 1>,
                memref<13824x160x16xui8, 1>)
        scales(%Sg, %Su : memref<13824x160xf16, 1>, memref<13824x160xf16, 1>)
        outs(%y : memref<1x?x13824xf16, 1>)
        {K = 5120 : i64, N = [13824, 13824], bits = 4 : i64,
         block_size = 32 : i64, accuracy_level = 4 : i64, epilogue = 1 : i64}
    return
  }

  // The missing segments are null weights, scales and outputs.
  // CHECK-LABEL: llvm.func @group_silu
  // CHECK:       %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK:       llvm.call @wrap_matmul_nbits_group(
  // CHECK-SAME:  %[[NULL]]

  func.func @group_quantized(%ctx: !hip.context,
                             %A: memref<1x1x64xf16, 1>,
                             %Bq: memref<64x2x16xui8, 1>,
                             %Bk: memref<32x2x16xui8, 1>,
                             %Sq: memref<64x2xf16, 1>,
                             %Sk: memref<32x2xf16, 1>,
                             %Aq: memref<1x1x64xi8, 1>,
                             %As: memref<1x1x2xf32, 1>,
                             %q: memref<1x1x64xf16, 1>,
                             %k: memref<1x1x32xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
        weights(%Bq, %Bk : memref<64x2x16xui8, 1>, memref<32x2x16xui8, 1>)
        scales(%Sq, %Sk : memref<64x2xf16, 1>, memref<32x2xf16, 1>)
        quantized(%Aq, %As : memref<1x1x64xi8, 1>, memref<1x1x2xf32, 1>)
        outs(%q, %k : memref<1x1x64xf16, 1>, memref<1x1x32xf16, 1>)
        {K = 64 : i64, N = [64, 32], bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64}
    return
  }

  // The quantized row's pointers are the last two arguments. The only null
  // pointers are the absent third and fourth segments' weight, scale and
  // output, and the four absent zero points.
  // CHECK-LABEL: llvm.func @group_quantized
  // CHECK-COUNT-10: llvm.mlir.zero : !llvm.ptr
  // CHECK-NOT:   llvm.mlir.zero
  // CHECK:       %[[AQ:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
  // CHECK-NEXT:  llvm.extractvalue
  // CHECK-NEXT:  %[[AS:.*]] = llvm.addrspacecast %{{.*}} : !llvm.ptr<1> to !llvm.ptr
  // CHECK-NEXT:  llvm.call @wrap_matmul_nbits_group({{.*}}, %[[AQ]], %[[AS]]) :

  func.func @group_zero_points(%ctx: !hip.context,
                               %A: memref<1x1x64xf16, 1>,
                               %Bg: memref<64x2x16xui8, 1>,
                               %Bu: memref<64x2x16xui8, 1>,
                               %Sg: memref<64x2xf16, 1>,
                               %Su: memref<64x2xf16, 1>,
                               %Zg: memref<64xui8, 1>,
                               %Zu: memref<64xui8, 1>,
                               %y: memref<1x1x64xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
        weights(%Bg, %Bu : memref<64x2x16xui8, 1>, memref<64x2x16xui8, 1>)
        scales(%Sg, %Su : memref<64x2xf16, 1>, memref<64x2xf16, 1>)
        zero_points(%Zg, %Zu : memref<64xui8, 1>, memref<64xui8, 1>)
        outs(%y : memref<1x1x64xf16, 1>)
        {K = 64 : i64, N = [64, 64], bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, epilogue = 1 : i64}
    return
  }

  // Both segments' zero points are passed; the nulls are the absent third and
  // fourth segments (weights, scales, zero points), the three unused outputs
  // and quantized(...).
  // CHECK-LABEL: llvm.func @group_zero_points
  // CHECK-COUNT-11: llvm.mlir.zero : !llvm.ptr
  // CHECK-NOT:   llvm.mlir.zero
  // CHECK:       llvm.call @wrap_matmul_nbits_group(

  func.func @group_row_offsets(%ctx: !hip.context,
                               %A: memref<1x1x64xf16, 1>,
                               %B: memref<12x2x16xui8, 1>,
                               %S: memref<12x2xf16, 1>,
                               %Z: memref<12xui8, 1>,
                               %a: memref<1x1x8xf16, 1>,
                               %b: memref<1x1x4xf16, 1>) {
    hip.matmul_nbits_group(%ctx) ins(%A : memref<1x1x64xf16, 1>)
        weights(%B, %B : memref<12x2x16xui8, 1>, memref<12x2x16xui8, 1>)
        scales(%S, %S : memref<12x2xf16, 1>, memref<12x2xf16, 1>)
        zero_points(%Z, %Z : memref<12xui8, 1>, memref<12xui8, 1>)
        outs(%a, %b : memref<1x1x8xf16, 1>, memref<1x1x4xf16, 1>)
        {K = 64 : i64, N = [8, 4], bits = 4 : i64, block_size = 32 : i64,
         accuracy_level = 4 : i64, row_offsets = array<i64: 0, 8>}
    return
  }

  // The second member starts 8 rows in: 8 x 2 blocks x 16 bytes of weights,
  // 8 x 2 fp16 scales and 8 x 1 byte of packed zero points. The first member
  // (row 0) is passed as is.
  // CHECK-LABEL: llvm.func @group_row_offsets
  // CHECK:       llvm.getelementptr inbounds %{{.*}}[256] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:       llvm.getelementptr inbounds %{{.*}}[32] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:       llvm.getelementptr inbounds %{{.*}}[8] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK-NOT:   llvm.getelementptr
  // CHECK:       llvm.call @wrap_matmul_nbits_group(
}
