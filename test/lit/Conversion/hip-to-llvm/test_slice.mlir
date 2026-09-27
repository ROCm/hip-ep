// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.slice lowers to a single wrap_slice call with the full
// (data, starts, ends, axes-or-null, steps-or-null, output, shape arrays,
// counts, dtype, host starts/ends/axes/steps-or-null) parameter pack
// regardless of whether the optional axes / steps operands or the host-known
// values are present.

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

  // Case 3: host-known bounds, axes and steps. Each becomes a stack array
  // handed over in the trailing host_* slots.
  func.func @test_slice_host(%ctx: !hip.context,
                              %data: memref<?x?xi64, 1>,
                              %starts: memref<1xi64, 1>,
                              %ends: memref<1xi64, 1>,
                              %axes: memref<1xi64, 1>,
                              %steps: memref<1xi64, 1>,
                              %s: index, %e: index,
                              %output: memref<?x?xi64, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<?x?xi64, 1>, memref<1xi64, 1>, memref<1xi64, 1>)
        axes(%axes : memref<1xi64, 1>)
        steps(%steps : memref<1xi64, 1>)
        host_bounds(%s, %e)
        outs(%output : memref<?x?xi64, 1>)
        {host_axes = array<i64: 1>, host_steps = array<i64: 1>}
    return
  }
}

// Each variant emits the same 19-parameter wrap_slice signature
// (4 + 3 pointers, 2 i64 shape ranks, 4 i64 counts/dtype, 4 host pointers).

// CHECK-LABEL: llvm.func @test_slice_full
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @test_slice_minimal
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @test_slice_host
// CHECK: %[[HS:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK: llvm.store
// CHECK: %[[HE:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK: llvm.store
// CHECK: %[[HA:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK: %[[HT:.*]] = llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK: llvm.call @wrap_slice({{.*}}, %[[HS]], %[[HE]], %[[HA]], %[[HT]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
