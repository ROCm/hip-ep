// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.gather_block_quantized lowers to wrap_gather_block_quantized
// with the full 21-parameter signature:
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
}

// CHECK-LABEL: llvm.func @test_gbq_static_with_zp
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gbq_static_no_zp
// CHECK: llvm.mlir.zero
// CHECK: llvm.call @wrap_gather_block_quantized({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gbq_dynamic_indices
// indices dim 0 read from descriptor — must be extractvalue, not a constant.
// CHECK: llvm.extractvalue %{{.*}}[3, 0]
// output dim 0 also dynamic — same pattern.
// CHECK: llvm.extractvalue %{{.*}}[3, 0]
// CHECK: llvm.call @wrap_gather_block_quantized
