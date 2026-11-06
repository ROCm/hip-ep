// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Integer outputs are not a normal distribution. The lowering must fail
// rather than dispatch them as a float kernel.
// RUN: not hip-mlir-opt --convert-hip-to-llvm %s

module {
  func.func @rnl_i32(
      %ctx: !hip.context,
      %input: memref<4xi32, 1>,
      %output: memref<4xi32, 1>) {
    hip.random_normal_like(%ctx)
        ins(%input : memref<4xi32, 1>)
        outs(%output : memref<4xi32, 1>)
        {seed = 1.0 : f32}
    return
  }
}
