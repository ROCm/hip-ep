// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.scatter_nd lowers to wrap_scatter_nd with the full
// 15-parameter signature: 5 pointers + 4 shape pointers + 4 ranks +
// reduction_id + data_type, regardless of the `reduction` attribute value.

module {
  func.func @test_scatter_nd_default(%ctx: !hip.context,
                                      %data: memref<4x4x4xf32, 1>,
                                      %indices: memref<2x1xi64, 1>,
                                      %updates: memref<2x4x4xf32, 1>,
                                      %output: memref<4x4x4xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates :
            memref<4x4x4xf32, 1>, memref<2x1xi64, 1>, memref<2x4x4xf32, 1>)
        outs(%output : memref<4x4x4xf32, 1>)
        {reduction = "none"}
    return
  }

  func.func @test_scatter_nd_add(%ctx: !hip.context,
                                  %data: memref<8xf32, 1>,
                                  %indices: memref<4x1xi64, 1>,
                                  %updates: memref<4xf32, 1>,
                                  %output: memref<8xf32, 1>) {
    hip.scatter_nd(%ctx)
        ins(%data, %indices, %updates :
            memref<8xf32, 1>, memref<4x1xi64, 1>, memref<4xf32, 1>)
        outs(%output : memref<8xf32, 1>)
        {reduction = "add"}
    return
  }
}

// CHECK-LABEL: llvm.func @test_scatter_nd_default
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_scatter_nd_add
// CHECK: llvm.call @wrap_scatter_nd({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
