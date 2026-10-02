// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

// Verify that hip.slice lowers to a single wrap_slice call with the full
// device/host pointer pairs for starts, ends, axes and steps, followed by the
// output, shape arrays, counts and dtype.

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

  func.func @test_slice_host_constants(%ctx: !hip.context,
                                        %data: memref<6xf32, 1>,
                                        %starts: memref<1xi64, 1>,
                                        %ends: memref<1xi64, 1>,
                                        %axes: memref<1xi64, 1>,
                                        %steps: memref<1xi64, 1>,
                                        %output: memref<3xf32, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<6xf32, 1>, memref<1xi64, 1>, memref<1xi64, 1>)
        axes(%axes : memref<1xi64, 1>)
        steps(%steps : memref<1xi64, 1>)
        outs(%output : memref<3xf32, 1>)
        {axes_attr = array<i64: 0>, ends_attr = array<i64: -1>,
         starts_attr = array<i64: 5>, steps_attr = array<i64: -2>}

    return
  }
}

// Each variant emits the same 19-parameter wrap_slice signature.

// CHECK-LABEL: llvm.func @test_slice_full
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_slice_minimal
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

// CHECK-LABEL: llvm.func @test_slice_host_constants
// CHECK: llvm.alloca {{.*}} x !llvm.array<1 x i64>
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
