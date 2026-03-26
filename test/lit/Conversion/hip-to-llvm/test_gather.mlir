// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @test_gather_lowering(%ctx: !hip.context,
                                   %data: memref<3x4xf32, 1>,
                                   %indices: memref<2xi64, 1>,
                                   %output: memref<2x4xf32, 1>) {
    hip.gather(%ctx)
        ins(%data, %indices : memref<3x4xf32, 1>, memref<2xi64, 1>)
        outs(%output : memref<2x4xf32, 1>)
        {axis = 0 : i64}
    return
  }
}

// CHECK-LABEL: llvm.func @test_gather_lowering
// CHECK: llvm.call @wrap_gather({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32

// Verify 9 parameters:
// - 4 pointers: state, data, indices, output
// - 5 i64 values: axis=0, data_num_elements=12, indices_num_elements=2, output_num_elements=8, element_size_bytes=4
