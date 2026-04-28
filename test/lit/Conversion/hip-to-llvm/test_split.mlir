// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @test_split
  // CHECK-SAME:  (%[[CTX:.*]]: !llvm.ptr
  func.func @test_split(%ctx: !hip.context, %input: memref<1x4xf32, 1>) -> memref<1x2xf32, 1> {
    %out = memref.alloc() : memref<1x2xf32, 1>
    // CHECK: llvm.call @wrap_split
    hip.split(%ctx) ins(%input : memref<1x4xf32, 1>)
                   outs(%out : memref<1x2xf32, 1>)
                   {axis = 1 : i64, offset = 2 : i64}
    return %out : memref<1x2xf32, 1>
  }
}
