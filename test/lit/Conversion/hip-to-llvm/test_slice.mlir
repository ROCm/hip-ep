// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.slice lowers to a single wrap_slice_typed call with the full
// (data, starts, ends, axes-or-null, steps-or-null, output, shape arrays,
// counts, dtype) parameter pack regardless of whether the optional axes /
// steps operands are present.

module {
  // Case 1: all five inputs present (data, starts, ends, axes, steps).
  func.func @test_slice_full(%ctx: !hip.context,
                              %data: memref<4x6xf32, 1>,
                              %starts: memref<2xi64, 1>,
                              %ends: memref<2xi64, 1>,
                              %axes: memref<2xi64, 1>,
                              %steps: memref<2xi64, 1>,
                              %output: memref<2x3xf32, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<4x6xf32, 1>, memref<2xi64, 1>, memref<2xi64, 1>)
        axes(%axes : memref<2xi64, 1>)
        steps(%steps : memref<2xi64, 1>)
        outs(%output : memref<2x3xf32, 1>)
    return
  }

  // Case 2: only the required inputs (data, starts, ends, output).
  func.func @test_slice_minimal(%ctx: !hip.context,
                                 %data: memref<8xf32, 1>,
                                 %starts: memref<1xi64, 1>,
                                 %ends: memref<1xi64, 1>,
                                 %output: memref<4xf32, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<8xf32, 1>, memref<1xi64, 1>, memref<1xi64, 1>)
        outs(%output : memref<4xf32, 1>)
    return
  }

  func.func @test_slice_i32(%ctx: !hip.context,
                            %data: memref<4x6xf32, 1>,
                            %starts: memref<2xi32, 1>,
                            %ends: memref<2xi32, 1>,
                            %axes: memref<2xi32, 1>,
                            %steps: memref<2xi32, 1>,
                            %output: memref<2x3xf32, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<4x6xf32, 1>, memref<2xi32, 1>, memref<2xi32, 1>)
        axes(%axes : memref<2xi32, 1>)
        steps(%steps : memref<2xi32, 1>)
        outs(%output : memref<2x3xf32, 1>)
    return
  }

  // Resolved compile-time parameters: the per-axis starts and steps go in
  // host stack arrays and the index tensors are never read.
  func.func @test_slice_static(%ctx: !hip.context,
                               %data: memref<4x6xf32, 1>,
                               %starts: memref<1xi64, 1>,
                               %ends: memref<1xi64, 1>,
                               %axes: memref<1xi64, 1>,
                               %steps: memref<1xi64, 1>,
                               %output: memref<4x6xf32, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<4x6xf32, 1>, memref<1xi64, 1>, memref<1xi64, 1>)
        axes(%axes : memref<1xi64, 1>)
        steps(%steps : memref<1xi64, 1>)
        outs(%output : memref<4x6xf32, 1>)
        {static_starts = array<i64: 0, 5>, static_steps = array<i64: 1, -1>}
    return
  }
}

// Each variant emits the same 16-parameter wrap_slice_typed signature
// (4 + 3 pointers, 2 i64 shape ranks, 4 i64 counts/dtype, index dtype).

// CHECK-LABEL: llvm.func @test_slice_full
// CHECK: llvm.call @wrap_slice_typed({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_slice_minimal
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_slice_typed({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_slice_i32
// FLOAT is 0. The constant created immediately after it is the index dtype.
// INT32 is 3, which is also the output extent, so a bare constant(3) would
// still match if the call passed INT64 (4).
// CHECK: %[[DATA_TYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INDEX_TYPE:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.call @wrap_slice_typed({{.*}}, %[[DATA_TYPE]], %[[INDEX_TYPE]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_slice_static
// CHECK-NOT: @wrap_slice_typed
// CHECK: %[[FIVE:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-NEXT: llvm.store %[[FIVE]]
// CHECK: %[[NEG:.*]] = llvm.mlir.constant(-1 : i64) : i64
// CHECK-NEXT: llvm.store %[[NEG]]
// CHECK: llvm.call @wrap_slice_static({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
