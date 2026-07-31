// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// `quant_storage_bits` records the width of one stored ONNX T1 element: 4 for
// tensor(int4)/tensor(uint4), 8 for tensor(uint8). Only those two widths are
// legal, and the storage element can never be narrower than the quantized
// value it holds.

// CHECK-LABEL: func.func @gbq_storage_bits_4
// CHECK:         hip.gather_block_quantized
func.func @gbq_storage_bits_4(%ctx: !hip.context,
                              %data: memref<2048x96xui8, 1>,
                              %indices: memref<8xi64, 1>,
                              %scales: memref<2048x12xf16, 1>,
                              %output: memref<8x96xf16, 1>) {
  hip.gather_block_quantized(%ctx)
      ins(%data, %indices, %scales :
          memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
      outs(%output : memref<8x96xf16, 1>)
      {bits = 4 : i64, block_size = 16 : i64,
       gather_axis = 0 : i64, quantize_axis = 1 : i64,
       quant_storage_bits = 4 : i64}
  return
}

// -----

// uint8 storage holding packed nibbles: storage width 8, quantized width 4.
// CHECK-LABEL: func.func @gbq_storage_bits_8_with_bits_4
// CHECK:         hip.gather_block_quantized
func.func @gbq_storage_bits_8_with_bits_4(%ctx: !hip.context,
                                          %data: memref<2048x96xui8, 1>,
                                          %indices: memref<8xi64, 1>,
                                          %scales: memref<2048x12xf16, 1>,
                                          %output: memref<8x96xf16, 1>) {
  hip.gather_block_quantized(%ctx)
      ins(%data, %indices, %scales :
          memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
      outs(%output : memref<8x96xf16, 1>)
      {bits = 4 : i64, block_size = 16 : i64,
       gather_axis = 0 : i64, quantize_axis = 1 : i64,
       unsigned_quant_storage, quant_storage_bits = 8 : i64}
  return
}

// -----

// Omitting the attribute is legal; the lowering falls back to `bits`.
// CHECK-LABEL: func.func @gbq_storage_bits_absent
// CHECK:         hip.gather_block_quantized
func.func @gbq_storage_bits_absent(%ctx: !hip.context,
                                   %data: memref<2048x96xui8, 1>,
                                   %indices: memref<8xi64, 1>,
                                   %scales: memref<2048x12xf16, 1>,
                                   %output: memref<8x96xf16, 1>) {
  hip.gather_block_quantized(%ctx)
      ins(%data, %indices, %scales :
          memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
      outs(%output : memref<8x96xf16, 1>)
      {bits = 4 : i64, block_size = 16 : i64,
       gather_axis = 0 : i64, quantize_axis = 1 : i64}
  return
}

// -----

func.func @gbq_storage_bits_illegal_width(%ctx: !hip.context,
                                          %data: memref<2048x96xui8, 1>,
                                          %indices: memref<8xi64, 1>,
                                          %scales: memref<2048x12xf16, 1>,
                                          %output: memref<8x96xf16, 1>) {
  // expected-error @+1 {{quant_storage_bits must be 4 or 8, got 3}}
  hip.gather_block_quantized(%ctx)
      ins(%data, %indices, %scales :
          memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
      outs(%output : memref<8x96xf16, 1>)
      {bits = 4 : i64, block_size = 16 : i64,
       gather_axis = 0 : i64, quantize_axis = 1 : i64,
       quant_storage_bits = 3 : i64}
  return
}

// -----

// A 4-bit storage element cannot hold an 8-bit quantized value.
func.func @gbq_storage_narrower_than_bits(%ctx: !hip.context,
                                          %data: memref<2048x96xui8, 1>,
                                          %indices: memref<8xi64, 1>,
                                          %scales: memref<2048x12xf16, 1>,
                                          %output: memref<8x96xf16, 1>) {
  // expected-error @+1 {{quant_storage_bits (4) must be >= bits (8)}}
  hip.gather_block_quantized(%ctx)
      ins(%data, %indices, %scales :
          memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
      outs(%output : memref<8x96xf16, 1>)
      {bits = 8 : i64, block_size = 16 : i64,
       gather_axis = 0 : i64, quantize_axis = 1 : i64,
       quant_storage_bits = 4 : i64}
  return
}
