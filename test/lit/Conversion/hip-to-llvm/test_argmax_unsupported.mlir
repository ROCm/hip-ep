// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ui32 and ui64 must not lower. getHipdnnDataType treats every 32/64-bit
// integer as signed, and the kernel would then rank a high bit as negative.
// RUN: not hip-mlir-opt --convert-hip-to-llvm %s

module {
  func.func @argmax_ui32(
      %ctx: !hip.context,
      %data: memref<2x3xui32, 1>,
      %indices: memref<2xi64, 1>) {
    hip.arg_max(%ctx)
        ins(%data : memref<2x3xui32, 1>)
        outs(%indices : memref<2xi64, 1>)
        {axis = 1 : i64, keepdims = 0 : i64, select_last_index = 0 : i64}
    return
  }

  func.func @argmax_ui64(
      %ctx: !hip.context,
      %data: memref<2x3xui64, 1>,
      %indices: memref<2xi64, 1>) {
    hip.arg_max(%ctx)
        ins(%data : memref<2x3xui64, 1>)
        outs(%indices : memref<2xi64, 1>)
        {axis = 1 : i64, keepdims = 0 : i64, select_last_index = 0 : i64}
    return
  }
}
