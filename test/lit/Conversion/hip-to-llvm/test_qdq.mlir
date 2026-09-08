// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.quantize_linear / hip.dequantize_linear lower to
// wrap_quantize_linear / wrap_dequantize_linear with the exact runtime ABI.
//
// wrap_quantize_linear (17 params):
//   state, input, scale, zero_point (nullable), output      (5 ptrs)
//   input_shape/rank, scale_shape/rank                      (2 * (ptr, i64))
//   axis, block_size                                        (2 i64)
//   precision, saturate                                     (2 i64)
//   input_dtype, scale_dtype, output_dtype                  (3 i64)
//   output_bits                                             (1 i64)
//
// wrap_dequantize_linear drops precision / saturate, which ONNX defines only
// on QuantizeLinear, and carries input_bits in place of output_bits
// (15 params).
//
// The three dtypes are HIPDNN_EP_DATATYPE_* values derived from the memref
// element types. They subsume the ONNX output_dtype attribute, which the
// importer already encoded in the type (UINT8 -> ui8, INT8 -> i8, and so on),
// so no TensorProto enum crosses this boundary. INT4 / UINT4 is the one type
// they cannot express, since both import as an 8-bit type; the trailing bit
// width carries it for whichever side is quantized.
//
// Output is elementwise-same-shape as input, so only input_shape and
// scale_shape are materialised.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file --verify-diagnostics %s | FileCheck %s

// ===== 1: QuantizeLinear, blocked along axis 1, with zero_point =====
// The ui8 output must reach the runtime as HIPDNN_EP_DATATYPE_UINT8 = 7, not
// as INT8 = 5. precision = 16 pins the attribute only QuantizeLinear carries.
// Without packed_int4, output_bits restates the storage width as 8.

// CHECK-LABEL: llvm.func @quantize_linear_blocked
// CHECK-DAG:   llvm.mlir.constant(16 : i64)
// CHECK-DAG:   llvm.mlir.constant(7 : i64)
// CHECK-DAG:   llvm.mlir.constant(8 : i64)
// CHECK:       llvm.call @wrap_quantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @quantize_linear_blocked(%ctx: !hip.context,
                                   %x: memref<1x64x32xf32, 1>,
                                   %scale: memref<1x2x32xf32, 1>,
                                   %zp: memref<1x2x32xui8, 1>,
                                   %y: memref<1x64x32xui8, 1>) {
  hip.quantize_linear(%ctx)
      ins(%x, %scale : memref<1x64x32xf32, 1>, memref<1x2x32xf32, 1>)
      zero_point(%zp : memref<1x2x32xui8, 1>)
      outs(%y : memref<1x64x32xui8, 1>)
      {axis = 1 : i64, block_size = 32 : i64, precision = 16 : i64,
       saturate = 0 : i64}
  return
}

// -----

// ===== 2: DequantizeLinear, per-axis, zero_point omitted =====
// An absent zero_point lowers to llvm.mlir.zero and leaves the signature
// unchanged. The signless i16 input is ONNX INT16, so it must reach the
// runtime as HIPDNN_EP_DATATYPE_INT16 = 8; no dimension is 8, so that
// constant can only come from input_dtype. Without packed_int4, input_bits
// restates the storage width, so it is 16 here.

// CHECK-LABEL: llvm.func @dequantize_linear_no_zp
// CHECK-DAG:   llvm.mlir.zero
// CHECK-DAG:   llvm.mlir.constant(8 : i64)
// CHECK-DAG:   llvm.mlir.constant(16 : i64)
// CHECK:       llvm.call @wrap_dequantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @dequantize_linear_no_zp(%ctx: !hip.context,
                                   %x: memref<6x128xi16, 1>,
                                   %scale: memref<128xf16, 1>,
                                   %y: memref<6x128xf16, 1>) {
  hip.dequantize_linear(%ctx)
      ins(%x, %scale : memref<6x128xi16, 1>, memref<128xf16, 1>)
      outs(%y : memref<6x128xf16, 1>)
      {axis = 1 : i64, block_size = 0 : i64}
  return
}

// -----

// ===== 3: QuantizeLinear rejected -- i32 is not a quantized storage type =====
// Only i8/ui8/i16/ui16 are accepted; a wider integer must fail to legalize
// rather than reach the runtime as HIPDNN_EP_DATATYPE_INT32.

func.func @quantize_linear_bad_storage(%ctx: !hip.context,
                                       %x: memref<8x128xf32, 1>,
                                       %scale: memref<f32, 1>,
                                       %y: memref<8x128xi32, 1>) {
  // expected-error @below {{failed to legalize operation 'hip.quantize_linear'}}
  hip.quantize_linear(%ctx)
      ins(%x, %scale : memref<8x128xf32, 1>, memref<f32, 1>)
      outs(%y : memref<8x128xi32, 1>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64}
  return
}

// -----

// ===== 4: DequantizeLinear rejected -- f64 is not a supported float side =====

func.func @dequantize_linear_bad_float(%ctx: !hip.context,
                                       %x: memref<8x128xi8, 1>,
                                       %scale: memref<f64, 1>,
                                       %y: memref<8x128xf64, 1>) {
  // expected-error @below {{failed to legalize operation 'hip.dequantize_linear'}}
  hip.dequantize_linear(%ctx)
      ins(%x, %scale : memref<8x128xi8, 1>, memref<f64, 1>)
      outs(%y : memref<8x128xf64, 1>)
      {axis = 1 : i64, block_size = 0 : i64}
  return
}

// -----

// ===== 5: DequantizeLinear, packed INT4, blocked with a packed zero_point ====
// packed_int4 changes exactly one thing in the ABI: input_bits becomes 4. The
// input keeps its i8 element type, so input_dtype stays INT8 = 5, and every
// shape stays logical -- 4096 elements on the block axis, not the 2048 bytes
// actually backing them. The zero_point rides the same flag; it is packed too
// and needs no parameter of its own.

// CHECK-LABEL: llvm.func @dequantize_linear_packed_int4
// CHECK-DAG:   llvm.mlir.constant(4096 : i64)
// CHECK-DAG:   llvm.mlir.constant(5 : i64)
// CHECK-DAG:   llvm.mlir.constant(4 : i64)
// CHECK:       llvm.call @wrap_dequantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @dequantize_linear_packed_int4(%ctx: !hip.context,
                                         %x: memref<4096x64xi8, 1>,
                                         %scale: memref<128x64xf16, 1>,
                                         %zp: memref<128x64xi8, 1>,
                                         %y: memref<4096x64xf16, 1>) {
  hip.dequantize_linear(%ctx)
      ins(%x, %scale : memref<4096x64xi8, 1>, memref<128x64xf16, 1>)
      zero_point(%zp : memref<128x64xi8, 1>)
      outs(%y : memref<4096x64xf16, 1>)
      {axis = 0 : i64, block_size = 32 : i64, packed_int4}
  return
}

// -----

// ===== 6: DequantizeLinear, packed UINT4 =====
// UINT4 imports as ui8, so input_dtype is UINT8 = 7 while input_bits is 4.
// That pair is the only thing distinguishing it from case 5, and it is what
// tells the kernel to zero-extend each nibble instead of sign-extending it.

// CHECK-LABEL: llvm.func @dequantize_linear_packed_uint4
// CHECK-DAG:   llvm.mlir.constant(7 : i64)
// CHECK-DAG:   llvm.mlir.constant(4 : i64)
// CHECK:       llvm.call @wrap_dequantize_linear
func.func @dequantize_linear_packed_uint4(%ctx: !hip.context,
                                          %x: memref<32x256xui8, 1>,
                                          %scale: memref<256xf32, 1>,
                                          %y: memref<32x256xf32, 1>) {
  hip.dequantize_linear(%ctx)
      ins(%x, %scale : memref<32x256xui8, 1>, memref<256xf32, 1>)
      outs(%y : memref<32x256xf32, 1>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4}
  return
}

// -----

// ===== 7: packed_int4 rejected on 16-bit storage =====
// Nothing packs two 4-bit values into an i16, so the flag can only mean the
// producer and this lowering disagree. Fail rather than emit input_bits = 4
// against a 16-bit stride.

func.func @dequantize_linear_packed_int4_bad_width(%ctx: !hip.context,
                                                   %x: memref<8x128xi16, 1>,
                                                   %scale: memref<f32, 1>,
                                                   %y: memref<8x128xf32, 1>) {
  // expected-error @below {{failed to legalize operation 'hip.dequantize_linear'}}
  hip.dequantize_linear(%ctx)
      ins(%x, %scale : memref<8x128xi16, 1>, memref<f32, 1>)
      outs(%y : memref<8x128xf32, 1>)
      {axis = 1 : i64, block_size = 0 : i64, packed_int4}
  return
}

// -----

// ===== 8: QuantizeLinear, packed UINT4 =====
// The mirror of case 6 on the quantize side: output_bits becomes 4 while the
// ui8 output type still reaches the runtime as UINT8 = 7, and the shapes stay
// logical even though only half the output buffer is written.

// CHECK-LABEL: llvm.func @quantize_linear_packed_uint4
// CHECK-DAG:   llvm.mlir.constant(7 : i64)
// CHECK-DAG:   llvm.mlir.constant(4 : i64)
// CHECK:       llvm.call @wrap_quantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
func.func @quantize_linear_packed_uint4(%ctx: !hip.context,
                                        %x: memref<32x256xf32, 1>,
                                        %scale: memref<256xf32, 1>,
                                        %zp: memref<256xui8, 1>,
                                        %y: memref<32x256xui8, 1>) {
  hip.quantize_linear(%ctx)
      ins(%x, %scale : memref<32x256xf32, 1>, memref<256xf32, 1>)
      zero_point(%zp : memref<256xui8, 1>)
      outs(%y : memref<32x256xui8, 1>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64, packed_int4}
  return
}

// -----

// ===== 9: packed_int4 rejected on a 16-bit quantize target =====

func.func @quantize_linear_packed_int4_bad_width(%ctx: !hip.context,
                                                 %x: memref<8x128xf32, 1>,
                                                 %scale: memref<f32, 1>,
                                                 %y: memref<8x128xi16, 1>) {
  // expected-error @below {{failed to legalize operation 'hip.quantize_linear'}}
  hip.quantize_linear(%ctx)
      ins(%x, %scale : memref<8x128xf32, 1>, memref<f32, 1>)
      outs(%y : memref<8x128xi16, 1>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64, packed_int4}
  return
}
