// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_slice(%ctx: !hip.context,
                        %data: memref<4x6xf32, 1>,
                        %valid: i1,
                        %s0: i64, %s1: i64,
                        %p0: i64, %p1: i64,
                        %e0: index, %e1: index,
                        %output: memref<2x3xf32, 1>) {
    hip.slice(%ctx) ins(%data : memref<4x6xf32, 1>)
        valid(%valid)
        starts(%s0, %s1 : i64, i64)
        steps(%p0, %p1 : i64, i64)
        extents(%e0, %e1 : index, index)
        outs(%output : memref<2x3xf32, 1>)
    return
  }
}

// CHECK-LABEL: llvm.func @test_slice
// CHECK: llvm.alloca
// CHECK: llvm.store %{{.*}} : i64, !llvm.ptr
// CHECK: llvm.call @wrap_slice({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i1) -> i32
