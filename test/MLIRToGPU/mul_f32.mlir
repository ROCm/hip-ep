// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Pre-bufferized memref IR for hip-compiler.
// This is the output of `hip-mlir-opt --onnx-to-hip-pipeline` on a simple
// onnx.Mul with two runtime tensor<8xf32> inputs.
//
// hip-compiler runs hip-to-llvm-pipeline internally, then compiles to DLL.
// The exported function `mul` uses the unpacked memref calling convention:
//   5 scalar args per rank-1 memref (alloc_ptr, align_ptr, offset, size0, stride0)
//   3 memrefs (A, B, C) => 15 total args.

module {
  func.func @mul(%arg0: memref<8xf32, strided<[?], offset: ?>>,
                 %arg1: memref<8xf32, strided<[?], offset: ?>>,
                 %arg2: memref<8xf32> {bufferize.result}) {
    %0 = hip.create_handle() : !hip.handle
    hip.miopen.mul(%0) ins(%arg0, %arg1 : memref<8xf32, strided<[?], offset: ?>>,
                                          memref<8xf32, strided<[?], offset: ?>>)
                       outs(%arg2 : memref<8xf32>)
    hip.destroy_handle(%0) : !hip.handle
    return
  }
}
