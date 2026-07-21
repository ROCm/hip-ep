// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s

module {
  func.func @compress_axis0_f32(
      %ctx: !hip.context,
      %input: memref<3x2xf32, 1>,
      %condition: memref<3xi1, 1>,
      %output: memref<2x2xf32, 1>) {
    // CHECK-LABEL: llvm.func @compress_axis0_f32

    hip.compress(%ctx)
        ins(%input, %condition : memref<3x2xf32, 1>, memref<3xi1, 1>)
        outs(%output : memref<2x2xf32, 1>) {axis = 0 : i64, flatten = false}

    // CHECK: llvm.call @wrap_compress({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    return
  }
}
