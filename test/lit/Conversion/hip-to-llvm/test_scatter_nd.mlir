// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.scatter_nd lowers to wrap_scatter_nd with the full
// 16-parameter signature: 6 pointers (state, data, indices, updates, output,
// count_ptr) + 4 shape pointers + 4 ranks + reduction_id + data_type. When
// has_valid_count=false the count_ptr arg is a null pointer (llvm.mlir.zero);
// when true it is the valid_count buffer pointer.

module {
  // has_valid_count=false -> count_ptr is null.
  func.func @test_scatter_nd_default(%ctx: !hip.context,
                                      %data: memref<4x4x4xf32, 1>,
                                      %indices: memref<2x1xi64, 1>,
                                      %updates: memref<2x4x4xf32, 1>,
                                      %count: memref<1xi32, 1>,
                                      %output: memref<4x4x4xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates, %count :
            memref<4x4x4xf32, 1>, memref<2x1xi64, 1>, memref<2x4x4xf32, 1>,
            memref<1xi32, 1>)
        outs(%output : memref<4x4x4xf32, 1>)
        {reduction = "none", has_valid_count = false}
    return
  }

  func.func @test_scatter_nd_add(%ctx: !hip.context,
                                  %data: memref<8xf32, 1>,
                                  %indices: memref<4x1xi64, 1>,
                                  %updates: memref<4xf32, 1>,
                                  %count: memref<1xi32, 1>,
                                  %output: memref<8xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates, %count :
            memref<8xf32, 1>, memref<4x1xi64, 1>, memref<4xf32, 1>,
            memref<1xi32, 1>)
        outs(%output : memref<8xf32, 1>)
        {reduction = "add", has_valid_count = false}
    return
  }

  // has_valid_count=true -> count_ptr is the valid_count buffer pointer.
  func.func @test_scatter_nd_valid_count(%ctx: !hip.context,
                                          %data: memref<8xf32, 1>,
                                          %indices: memref<4x1xi64, 1>,
                                          %updates: memref<4xf32, 1>,
                                          %count: memref<1xi32, 1>,
                                          %output: memref<8xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates, %count :
            memref<8xf32, 1>, memref<4x1xi64, 1>, memref<4xf32, 1>,
            memref<1xi32, 1>)
        outs(%output : memref<8xf32, 1>)
        {reduction = "none", has_valid_count = true}
    return
  }
}

// CHECK-LABEL: llvm.func @test_scatter_nd_default
// has_valid_count=false materialises a null count pointer.
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_scatter_nd_add
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_scatter_nd_valid_count
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
