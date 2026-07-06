// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  // Multi-index gather: indices has >1 element (embedding lookup pattern)
  // data=[3x4xf32], indices=[2xi64] -> output=[2x4xf32]
  // data_num=12, indices_num=2, output_num=8, elem_size=4
  func.func @test_gather_multi_index(%ctx: !hip.context,
                                      %data: memref<3x4xf32, 1>,
                                      %indices: memref<2xi64, 1>,
                                      %output: memref<2x4xf32, 1>) {
    hip.gather(%ctx)
        ins(%data, %indices : memref<3x4xf32, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xf32, 1>)
        {axis = 0 : i64}
    return
  }

  // Scalar-index gather: indices has exactly 1 element
  // data=[2xi64], indices=[1xi64] -> output=[1xi64]
  // data_num=2, indices_num=1, output_num=1, elem_size=8
  func.func @test_gather_scalar_index(%ctx: !hip.context,
                                       %data: memref<2xi64, 1>,
                                       %indices: memref<1xi64, 1>,
                                       %output: memref<1xi64, 1>) {
    hip.gather(%ctx)
        ins(%data, %indices : memref<2xi64, 1>, memref<1xi64, 1>)
        outs(%output : memref<1xi64, 1>)
        {axis = 0 : i64}
    return
  }

  // Embedding-shaped gather: realistic LLM vocab lookup
  // data=[32000x128xf16], indices=[128xi64] -> output=[128x128xf16]
  // data_num=4096000, indices_num=128, output_num=16384, elem_size=2
  func.func @test_gather_embedding(%ctx: !hip.context,
                                    %data: memref<32000x128xf16, 1>,
                                    %indices: memref<128xi64, 1>,
                                    %output: memref<128x128xf16, 1>) {
    hip.gather(%ctx)
        ins(%data, %indices : memref<32000x128xf16, 1>, memref<128xi64, 1>)
        outs(%output : memref<128x128xf16, 1>)
        {axis = 0 : i64}
    return
  }
}

// All three variants lower to the same wrap_gather call:
// 4 pointers + 16 i64 (axis, counts, axis_size, inner, elem sizes,
// 3 shape ptrs + 3 ranks, 2 dtype ids) — see GatherLowering.cpp.

// CHECK-LABEL: llvm.func @test_gather_multi_index
// CHECK: llvm.call @wrap_gather({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gather_scalar_index
// CHECK: llvm.call @wrap_gather({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_gather_embedding
// CHECK: llvm.call @wrap_gather({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
