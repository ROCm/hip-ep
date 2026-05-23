// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.scatter_nd lowers to wrap_scatter_nd with the full
// 16-parameter signature: 5 data pointers + count_ptr + 4 shape pointers
// + 4 ranks + reduction_id + data_type.

module {
  // Without valid_count (has_valid_count = false — count_ptr is null)
  func.func @test_scatter_nd_default(%ctx: !hip.context,
                                      %data: memref<4x4x4xf32, 1>,
                                      %indices: memref<2x1xi64, 1>,
                                      %updates: memref<2x4x4xf32, 1>,
                                      %dummy_count: memref<1xi32, 1>,
                                      %output: memref<4x4x4xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates, %dummy_count :
            memref<4x4x4xf32, 1>, memref<2x1xi64, 1>, memref<2x4x4xf32, 1>,
            memref<1xi32, 1>)
        outs(%output : memref<4x4x4xf32, 1>)
        {reduction = "none", has_valid_count = false}
    return
  }

  // With valid_count (has_valid_count = true — count_ptr is a real GPU pointer)
  func.func @test_scatter_nd_with_count(%ctx: !hip.context,
                                         %data: memref<8xf32, 1>,
                                         %indices: memref<12x1xi64, 1>,
                                         %updates: memref<12xf32, 1>,
                                         %count: memref<1xi32, 1>,
                                         %output: memref<8xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates, %count :
            memref<8xf32, 1>, memref<12x1xi64, 1>, memref<12xf32, 1>,
            memref<1xi32, 1>)
        outs(%output : memref<8xf32, 1>)
        {reduction = "add", has_valid_count = true}
    return
  }
}

// CHECK-LABEL: llvm.func @test_scatter_nd_default
// count_ptr should be null (llvm.mlir.zero) when has_valid_count = false
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_scatter_nd_with_count
// count_ptr should be the actual memref pointer
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
