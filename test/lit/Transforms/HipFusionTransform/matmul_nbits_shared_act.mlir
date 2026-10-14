// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// MatMulNBitsSharedActivation quantizes an activation read by two or more
// W4A8-eligible hip.matmul_nbits ops once, with hip.quantize_act, and hands
// the result to each reader through `quantized(...)`.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Two readers with the same block_size share one quantize op, placed right
// after the activation and sized from its dynamic dims.
// CHECK-LABEL: func.func @two_readers
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[M:.*]] = tensor.dim %arg1, %[[C1]] : tensor<1x?x64xf16>
// CHECK:         %[[EQ:.*]] = tensor.empty(%[[M]]) : tensor<1x?x64xi8>
// CHECK:         %[[ES:.*]] = tensor.empty(%[[M]]) : tensor<1x?x2xf32>
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act(%arg0) ins(%arg1 : tensor<1x?x64xf16>)
// CHECK-SAME:      outs(%[[EQ]], %[[ES]] : tensor<1x?x64xi8>, tensor<1x?x2xf32>)
// CHECK-SAME:      block_size = 32
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg2, %arg3 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg4, %arg5 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK-NOT:     hip.quantize_act
func.func @two_readers(%ctx: !hip.context, %x: tensor<1x?x64xf16>,
                       %bq: tensor<64x2x16xui8>, %sq: tensor<64x2xf16>,
                       %bk: tensor<32x2x16xui8>, %sk: tensor<32x2xf16>)
    -> (tensor<1x?x64xf16>, tensor<1x?x32xf16>) {
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
  return %q, %k : tensor<1x?x64xf16>, tensor<1x?x32xf16>
}

// -----

// A single reader keeps quantizing inside its own kernel.
// CHECK-LABEL: func.func @one_reader
// CHECK-NOT:     hip.quantize_act
// CHECK-NOT:     quantized(
func.func @one_reader(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                      %b: tensor<64x2x16xui8>, %s: tensor<64x2xf16>)
    -> tensor<1x1x64xf16> {
  %e = tensor.empty() : tensor<1x1x64xf16>
  %y = hip.matmul_nbits(%ctx) ins(%x, %b, %s :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      outs(%e : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  return %y : tensor<1x1x64xf16>
}

// -----

// Readers of one activation with different block sizes cannot share a
// quantization: the per-block scales differ. Only the pair at block_size 32
// gets one; the block_size 64 reader is left alone.
// CHECK-LABEL: func.func @mixed_block_size
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act(%arg0) ins(%arg1 : tensor<1x1x128xf16>)
// CHECK-SAME:      block_size = 32
// CHECK-NOT:     hip.quantize_act
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg2, %arg3 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg4, %arg5 :
// CHECK-NOT:       quantized(
// CHECK-SAME:      block_size = 64
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg6, %arg7 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
func.func @mixed_block_size(%ctx: !hip.context, %x: tensor<1x1x128xf16>,
                            %b0: tensor<64x4x16xui8>, %s0: tensor<64x4xf16>,
                            %b1: tensor<64x2x32xui8>, %s1: tensor<64x2xf16>,
                            %b2: tensor<64x4x16xui8>, %s2: tensor<64x4xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x128xf16>, tensor<64x4x16xui8>, tensor<64x4xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {K = 128 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x128xf16>, tensor<64x2x32xui8>, tensor<64x2xf16>)
      outs(%e1 : tensor<1x1x64xf16>)
      {K = 128 : i64, N = 64 : i64, bits = 4 : i64, block_size = 64 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x, %b2, %s2 :
      tensor<1x1x128xf16>, tensor<64x4x16xui8>, tensor<64x4xf16>)
      outs(%e2 : tensor<1x1x64xf16>)
      {K = 128 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  return %y0, %y1, %y2 : tensor<1x1x64xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// Asymmetric readers with packed uint8 zero points qualify; fp16 zero points
// do not (the dp4a path has no fp16 zero-point form).
// CHECK-LABEL: func.func @zero_points
// CHECK:         %[[Q:.*]]:2 = hip.quantize_act
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg2, %arg3 :
// CHECK-SAME:      zero_points(%arg4 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg5, %arg6 :
// CHECK-SAME:      zero_points(%arg7 :
// CHECK-SAME:      quantized(%[[Q]]#0, %[[Q]]#1 :
// CHECK:         hip.matmul_nbits(%arg0) ins(%arg1, %arg8, %arg9 :
// CHECK-NOT:       quantized(
// CHECK-SAME:      zp_elem_size = 2
func.func @zero_points(%ctx: !hip.context, %x: tensor<1x1x64xf16>,
                       %b0: tensor<64x2x16xui8>, %s0: tensor<64x2xf16>,
                       %z0: tensor<64xui8>,
                       %b1: tensor<64x2x16xui8>, %s1: tensor<64x2xf16>,
                       %z1: tensor<64xui8>,
                       %b2: tensor<64x2x16xui8>, %s2: tensor<64x2xf16>,
                       %z2: tensor<64x2xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      zero_points(%z0 : tensor<64xui8>)
      outs(%e0 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      zero_points(%z1 : tensor<64xui8>)
      outs(%e1 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 1 : i64} : tensor<1x1x64xf16>
  %e2 = tensor.empty() : tensor<1x1x64xf16>
  %y2 = hip.matmul_nbits(%ctx) ins(%x, %b2, %s2 :
      tensor<1x1x64xf16>, tensor<64x2x16xui8>, tensor<64x2xf16>)
      zero_points(%z2 : tensor<64x2xf16>)
      outs(%e2 : tensor<1x1x64xf16>)
      {K = 64 : i64, N = 64 : i64, bits = 4 : i64, block_size = 32 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 2 : i64} : tensor<1x1x64xf16>
  return %y0, %y1, %y2 : tensor<1x1x64xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>
}

// -----

// K not a multiple of 32 is off the dp4a path.
// CHECK-LABEL: func.func @k_not_multiple_of_32
// CHECK-NOT:     hip.quantize_act
func.func @k_not_multiple_of_32(%ctx: !hip.context, %x: tensor<1x1x48xf16>,
                                %b0: tensor<64x3x8xui8>, %s0: tensor<64x3xf16>,
                                %b1: tensor<64x3x8xui8>, %s1: tensor<64x3xf16>)
    -> (tensor<1x1x64xf16>, tensor<1x1x64xf16>) {
  %e0 = tensor.empty() : tensor<1x1x64xf16>
  %y0 = hip.matmul_nbits(%ctx) ins(%x, %b0, %s0 :
      tensor<1x1x48xf16>, tensor<64x3x8xui8>, tensor<64x3xf16>)
      outs(%e0 : tensor<1x1x64xf16>)
      {K = 48 : i64, N = 64 : i64, bits = 4 : i64, block_size = 16 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  %e1 = tensor.empty() : tensor<1x1x64xf16>
  %y1 = hip.matmul_nbits(%ctx) ins(%x, %b1, %s1 :
      tensor<1x1x48xf16>, tensor<64x3x8xui8>, tensor<64x3xf16>)
      outs(%e1 : tensor<1x1x64xf16>)
      {K = 48 : i64, N = 64 : i64, bits = 4 : i64, block_size = 16 : i64,
       accuracy_level = 4 : i64, zp_elem_size = 0 : i64} : tensor<1x1x64xf16>
  return %y0, %y1 : tensor<1x1x64xf16>, tensor<1x1x64xf16>
}
