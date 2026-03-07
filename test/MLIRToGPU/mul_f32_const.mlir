// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Pre-bufferized memref IR for hip-compiler.
// This is the output of:
//   hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=4'
// on an onnx.Mul where B is an onnx.Constant (8 non-splat f32 elements).
//
// The constant blob is passed as memref<?xi8> and viewed as memref<8xf32>.
// Exported function signature (unpacked): 15 args total
//   arg0-4:  A  (memref<8xf32>)        -- runtime input
//   arg5-9:  C  (memref<8xf32>)        -- output
//   arg10-14: constants blob (memref<?xi8>) -- contains B data

module {
  func.func @mul_const(%arg0: memref<8xf32, strided<[?], offset: ?>>,
                       %arg1: memref<8xf32> {bufferize.result},
                       %arg2: memref<?xi8>) {
    %0 = hip.create_handle() : !hip.handle
    %c0 = arith.constant 0 : index
    %view = memref.view %arg2[%c0][] : memref<?xi8> to memref<8xf32>
    hip.miopen.mul(%0) ins(%arg0, %view : memref<8xf32, strided<[?], offset: ?>>,
                                          memref<8xf32>)
                       outs(%arg1 : memref<8xf32>)
    hip.destroy_handle(%0) : !hip.handle
    return
  }
}
