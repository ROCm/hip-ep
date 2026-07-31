// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.gather_block_quantized lowers to wrap_gather_block_quantized
// with the full 22-parameter signature:
//   state, data, indices, scales, zero_points (nullable), output  (6 ptrs)
//   data_shape/rank, idx_shape/rank, scl_shape/rank, out_shape/rank
//                                                              (4 * (ptr, i64))
//   bits, block_size, gather_axis, quantize_axis            (4 * i64)
//   data_dtype, indices_dtype, scales_dtype                 (3 * i64)
// Regardless of whether zero_points is supplied, the runtime signature
// stays the same — absent zero_points lowers to llvm.mlir.zero.
//
// The dynamic-indices case additionally exercises the runtime shape array
// build: the dim 0 of `%indices` and dim 0 of `%output` must come from
// `llvm.extractvalue %md[3, 0]` (MemRef descriptor sizes field), not from a
// constant — using kDynamic sentinel as a runtime size would be UB.

module {
  // ===== Test 1: Static shapes, with zero_points =====

  func.func @test_gbq_static_with_zp(%ctx: !hip.context,
                                      %data: memref<2048x96xui8, 1>,
                                      %indices: memref<8xi64, 1>,
                                      %scales: memref<2048x12xf16, 1>,
                                      %zp: memref<2048x12xui8, 1>,
                                      %output: memref<8x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
        zero_points(%zp : memref<2048x12xui8, 1>)
        outs(%output : memref<8x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64}
    return
  }

  // ===== Test 2: Static shapes, no zero_points =====

  func.func @test_gbq_static_no_zp(%ctx: !hip.context,
                                    %data: memref<512x64xui8, 1>,
                                    %indices: memref<4xi32, 1>,
                                    %scales: memref<512x2xf32, 1>,
                                    %output: memref<4x64xf32, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<512x64xui8, 1>, memref<4xi32, 1>, memref<512x2xf32, 1>)
        outs(%output : memref<4x64xf32, 1>)
        {bits = 8 : i64, block_size = 32 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64}
    return
  }

  // ===== Test 3: Dynamic indices length =====

  func.func @test_gbq_dynamic_indices(%ctx: !hip.context,
                                       %data: memref<2048x96xui8, 1>,
                                       %indices: memref<?xi64, 1>,
                                       %scales: memref<2048x12xf16, 1>,
                                       %output: memref<?x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>, memref<?xi64, 1>, memref<2048x12xf16, 1>)
        outs(%output : memref<?x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64}
    return
  }

  // ===== Test 4: signless i8 + unsigned_quant_storage =====
  // unsigned_quant_storage marks UINT4 packed in signless i8; lowering must pass
  // HIPDNN_EP_DATATYPE_UINT8 (7), not signless-i8 default (5).

  func.func @test_gbq_unsigned_quant_storage(%ctx: !hip.context,
                                              %data: memref<2048x96xi8, 1>,
                                              %indices: memref<8xi64, 1>,
                                              %scales: memref<2048x12xf16, 1>,
                                              %output: memref<8x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xi8, 1>, memref<8xi64, 1>, memref<2048x12xf16, 1>)
        outs(%output : memref<8x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64,
         unsigned_quant_storage}
    return
  }

  // ===== Tests 5-7: quant_storage_bits reaches the runtime by value =====
  // The trailing runtime argument is the ONNX T1 storage width, which is NOT
  // `bits`: tensor(uint8) legally holds 4-bit values two per byte. The three
  // cases below are identical apart from that attribute, so checking the
  // argument list positionally is the only way to catch it being dropped,
  // reordered, or silently defaulted back to `bits`.
  //
  // Shapes deliberately avoid a literal 4 in any dim or rank so the first
  // `constant(4 : i64)` in each function is unambiguously `bits`.

  // Test 5: uint8 storage, 4-bit values -> storage width 8 while bits is 4.

  func.func @test_gbq_storage_bits_uint8_packed(%ctx: !hip.context,
                                                 %data: memref<2048x96xui8, 1>,
                                                 %indices: memref<5xi64, 1>,
                                                 %scales: memref<2048x12xf16, 1>,
                                                 %output: memref<5x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>, memref<5xi64, 1>, memref<2048x12xf16, 1>)
        outs(%output : memref<5x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64,
         unsigned_quant_storage, quant_storage_bits = 8 : i64}
    return
  }

  // Test 6: native uint4 storage -> storage width 4, same bits, same types.

  func.func @test_gbq_storage_bits_uint4(%ctx: !hip.context,
                                          %data: memref<2048x96xui8, 1>,
                                          %indices: memref<5xi64, 1>,
                                          %scales: memref<2048x12xf16, 1>,
                                          %output: memref<5x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>, memref<5xi64, 1>, memref<2048x12xf16, 1>)
        outs(%output : memref<5x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64,
         unsigned_quant_storage, quant_storage_bits = 4 : i64}
    return
  }

  // Test 7: attribute absent (hand-written IR, or a non-constant `data`) ->
  // falls back to `bits` rather than guessing.

  func.func @test_gbq_storage_bits_absent(%ctx: !hip.context,
                                           %data: memref<2048x96xui8, 1>,
                                           %indices: memref<5xi64, 1>,
                                           %scales: memref<2048x12xf16, 1>,
                                           %output: memref<5x96xf16, 1>) {
    hip.gather_block_quantized(%ctx)
        ins(%data, %indices, %scales :
            memref<2048x96xui8, 1>, memref<5xi64, 1>, memref<2048x12xf16, 1>)
        outs(%output : memref<5x96xf16, 1>)
        {bits = 4 : i64, block_size = 16 : i64,
         gather_axis = 0 : i64, quantize_axis = 1 : i64,
         unsigned_quant_storage}
    return
  }
}

// CHECK-LABEL: llvm.func @test_gbq_static_with_zp
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gbq_static_no_zp
// CHECK: llvm.mlir.zero
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gbq_dynamic_indices
// indices dim 0 read from descriptor — must be extractvalue, not a constant.
// CHECK: llvm.extractvalue %{{.*}}[3, 0]
// output dim 0 also dynamic — same pattern.
// CHECK: llvm.extractvalue %{{.*}}[3, 0]
// CHECK: llvm.call @wrap_gather_block_quantized

// CHECK-LABEL: llvm.func @test_gbq_unsigned_quant_storage
// data_dtype must be HIPDNN_EP_DATATYPE_UINT8 (7), not signless-i8 default (5).
// CHECK-DAG: llvm.mlir.constant(7 : i64) : i64
// CHECK-NOT: llvm.mlir.constant(5 : i64) : i64
// CHECK: llvm.call @wrap_gather_block_quantized

// The scalar tail is emitted in a fixed order immediately before the call:
// bits, block_size, gather_axis, quantize_axis, data_dtype, indices_dtype,
// scales_dtype, quant_storage_bits. Binding each to an SSA name and then
// requiring that exact sequence as the last eight call operands pins both the
// value and the position of the storage width.

// CHECK-LABEL: llvm.func @test_gbq_storage_bits_uint8_packed
// CHECK: %[[BITS:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[BLK:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK: %[[GA:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[QA:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DDT:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK: %[[IDT:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[SDT:.*]] = llvm.mlir.constant(1 : i64) : i64
// storage width 8 even though bits is 4 -- the whole point of the argument.
// CHECK: %[[QSB:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}, %[[BITS]], %[[BLK]], %[[GA]], %[[QA]], %[[DDT]], %[[IDT]], %[[SDT]], %[[QSB]])

// CHECK-LABEL: llvm.func @test_gbq_storage_bits_uint4
// CHECK: %[[BITS:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[BLK:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK: %[[GA:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[QA:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DDT:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK: %[[IDT:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[SDT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[QSB:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}, %[[BITS]], %[[BLK]], %[[GA]], %[[QA]], %[[DDT]], %[[IDT]], %[[SDT]], %[[QSB]])

// CHECK-LABEL: llvm.func @test_gbq_storage_bits_absent
// CHECK: %[[BITS:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[BLK:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK: %[[GA:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[QA:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DDT:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK: %[[IDT:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: %[[SDT:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[QSB:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}, %[[BITS]], %[[BLK]], %[[GA]], %[[QA]], %[[DDT]], %[[IDT]], %[[SDT]], %[[QSB]])
