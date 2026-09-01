// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.quantize_linear / hip.dequantize_linear lower to
// wrap_quantize_linear / wrap_dequantize_linear with the exact runtime ABI.
//
// wrap_quantize_linear (16 params):
//   state, input, scale, zero_point (nullable), output      (5 ptrs)
//   input_shape/rank, scale_shape/rank                      (2 * (ptr, i64))
//   axis, block_size                                        (2 i64)
//   precision, saturate                                     (2 i64)
//   input_dtype, scale_dtype, output_dtype                  (3 i64)
//
// wrap_dequantize_linear is the same list without precision / saturate
// (14 params): ONNX only defines those two on QuantizeLinear.
//
// The trailing three are HIPDNN_EP_DATATYPE_* values derived from the memref
// element types. They subsume the ONNX output_dtype attribute, which the
// importer already encoded in the type (UINT8 -> ui8, INT8 -> i8, and so on),
// so no TensorProto enum crosses this boundary.
//
// Output is elementwise-same-shape as input, so only input_shape and
// scale_shape are materialised.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file --verify-diagnostics %s | FileCheck %s

// ===== 1: QuantizeLinear, blocked along axis 1, with zero_point =====
// The ui8 output must reach the runtime as HIPDNN_EP_DATATYPE_UINT8 = 7, not
// as INT8 = 5. precision = 16 pins the attribute only QuantizeLinear carries.

// CHECK-LABEL: llvm.func @quantize_linear_blocked
// CHECK-DAG:   llvm.mlir.constant(16 : i64)
// CHECK-DAG:   llvm.mlir.constant(7 : i64)
// CHECK:       llvm.call @wrap_quantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
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
// constant can only come from input_dtype.

// CHECK-LABEL: llvm.func @dequantize_linear_no_zp
// CHECK-DAG:   llvm.mlir.zero
// CHECK-DAG:   llvm.mlir.constant(8 : i64)
// CHECK:       llvm.call @wrap_dequantize_linear({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32
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
