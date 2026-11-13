// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.unpack_int4 extracts low-nibble-first int4 values. A blocked
// hip.dequantize_linear broadcasts one scale and zero point per K-block,
// then subtracts and multiplies. block_size = 0 stays on the per-tensor path
// in test_qdq.mlir.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @unpack_int4
// CHECK: tosa.bitwise_and
// CHECK: tosa.logical_right_shift
// CHECK: tosa.concat
// CHECK-NOT: hip.unpack_int4
func.func @unpack_int4(%ctx: !hip.context, %packed: tensor<2x8xui8>,
                       %init: tensor<2x16xui8>) -> tensor<2x16xui8>
    attributes {rock.kernel} {
  %r = hip.unpack_int4(%ctx) ins(%packed : tensor<2x8xui8>)
      outs(%init : tensor<2x16xui8>) : tensor<2x16xui8>
  return %r : tensor<2x16xui8>
}

// -----

// The MatMulNBits default zero point is the constant 8. Keep the K-block
// dimension explicit so scale and zero point broadcast from [N, blocks, 1]
// rather than being expanded with tosa.tile, which rocMLIR cannot legalize.
// CHECK-LABEL: func.func @blocked_dequant_default_zp
// CHECK: tosa.const
// CHECK-SAME: dense<8.000000e+00>
// CHECK: tensor<2x1x16xf16>
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: tosa.tile
// CHECK-NOT: hip.dequantize_linear
func.func @blocked_dequant_default_zp(%ctx: !hip.context,
                                      %x: tensor<2x16xui8>,
                                      %scales: tensor<2x1xf16>,
                                      %init: tensor<2x16xf16>)
    -> tensor<2x16xf16> attributes {rock.kernel} {
  %zp = hip.constant {value = dense<8.000000e+00> : tensor<2x1xf16>}
      : tensor<2x1xf16>
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scales : tensor<2x16xui8>, tensor<2x1xf16>)
      zero_point(%zp : tensor<2x1xf16>)
      outs(%init : tensor<2x16xf16>)
      {axis = 1 : i64, block_size = 16 : i64} : tensor<2x16xf16>
  return %r : tensor<2x16xf16>
}

// -----

// CHECK-LABEL: func.func @blocked_dequant_explicit_zp
// CHECK: tensor<2x1x16xf16>
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: tosa.tile
// CHECK-NOT: hip.dequantize_linear
func.func @blocked_dequant_explicit_zp(%ctx: !hip.context,
                                       %x: tensor<2x16xui8>,
                                       %scales: tensor<2x1xf16>,
                                       %zp: tensor<2x1xf16>,
                                       %init: tensor<2x16xf16>)
    -> tensor<2x16xf16> attributes {rock.kernel} {
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scales : tensor<2x16xui8>, tensor<2x1xf16>)
      zero_point(%zp : tensor<2x1xf16>)
      outs(%init : tensor<2x16xf16>)
      {axis = 1 : i64, block_size = 16 : i64} : tensor<2x16xf16>
  return %r : tensor<2x16xf16>
}

// -----

// Packed MatMulNBits weights that are a kernel argument stay packed in memory
// and are presented as logical signless i4. rocMLIR extracts the nibbles. A
// ui8 bitwise unpack is what makes gridwise GEMM reject B. arith.uitofp does
// not accept ui4.
// CHECK-LABEL: func.func @blocked_dequant_logical_int4
// CHECK-SAME: tensor<2x16xi4>
// CHECK-NOT: tosa.bitwise_and
// CHECK-NOT: tosa.tile
// CHECK: tensor<2x1x16xi4>
// CHECK: unsigned_cast
// CHECK: tensor<2x1x16xf16>
// CHECK: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: hip.unpack_int4
func.func @blocked_dequant_logical_int4(
    %ctx: !hip.context, %packed: tensor<2x8xui8>, %scales: tensor<2x1xf16>,
    %unpacked: tensor<2x16xui8>, %init: tensor<2x16xf16>) -> tensor<2x16xf16>
    attributes {rock.kernel} {
  %bytes = hip.unpack_int4(%ctx) ins(%packed : tensor<2x8xui8>)
      outs(%unpacked : tensor<2x16xui8>) : tensor<2x16xui8>
  %zp = hip.constant {value = dense<8.000000e+00> : tensor<2x1xf16>}
      : tensor<2x1xf16>
  %r = hip.dequantize_linear(%ctx)
      ins(%bytes, %scales : tensor<2x16xui8>, tensor<2x1xf16>)
      zero_point(%zp : tensor<2x1xf16>)
      outs(%init : tensor<2x16xf16>)
      {axis = 1 : i64, block_size = 16 : i64} : tensor<2x16xf16>
  return %r : tensor<2x16xf16>
}

// -----

// No zero point is a multiply by the broadcast scale.
// CHECK-LABEL: func.func @blocked_dequant_no_zp
// CHECK: tensor<2x1x16xf16>
// CHECK-NOT: tosa.sub
// CHECK: tosa.mul
// CHECK-NOT: tosa.tile
// CHECK-NOT: hip.dequantize_linear
func.func @blocked_dequant_no_zp(%ctx: !hip.context, %x: tensor<2x16xui8>,
                                 %scales: tensor<2x1xf16>,
                                 %init: tensor<2x16xf16>)
    -> tensor<2x16xf16> attributes {rock.kernel} {
  %r = hip.dequantize_linear(%ctx)
      ins(%x, %scales : tensor<2x16xui8>, tensor<2x1xf16>)
      outs(%init : tensor<2x16xf16>)
      {axis = 1 : i64, block_size = 16 : i64} : tensor<2x16xf16>
  return %r : tensor<2x16xf16>
}
