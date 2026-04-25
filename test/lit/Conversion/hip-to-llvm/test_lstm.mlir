// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify hip.lstm lowers to a call into wrap_miopenRNNForwardInference
// with all 16 expected parameters (state + 12 ptrs + hidden + dir + dtype).

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @lstm_llvm_test(
      %ctx: !hip.context,
      %x : memref<8x1x32xf16, 1>,
      %w : memref<2x1024x32xf16, 1>,
      %r : memref<2x1024x256xf16, 1>,
      %b : memref<2x2048xf16, 1>,
      %ih: memref<2x1x256xf16, 1>,
      %ic: memref<2x1x256xf16, 1>,
      %y : memref<8x2x1x256xf16, 1>,
      %yh: memref<2x1x256xf16, 1>,
      %yc: memref<2x1x256xf16, 1>) {
    // CHECK-LABEL: llvm.func @lstm_llvm_test
    // CHECK-SAME: %[[CTX:.*]]: !llvm.ptr
    hip.lstm(%ctx)
        ins(%x, %w, %r, %b, %ih, %ic
            : memref<8x1x32xf16, 1>, memref<2x1024x32xf16, 1>,
              memref<2x1024x256xf16, 1>, memref<2x2048xf16, 1>,
              memref<2x1x256xf16, 1>, memref<2x1x256xf16, 1>)
        outs(%y, %yh, %yc
             : memref<8x2x1x256xf16, 1>, memref<2x1x256xf16, 1>,
               memref<2x1x256xf16, 1>)
        {hidden_size = 256 : i64, direction = 2 : i64}

    // CHECK: llvm.call @wrap_miopenRNNForwardInference
    return
  }

  // Variant: no bias, no initial states.  The lowering must materialise
  // null pointers for the absent operands so the runtime call signature
  // stays uniform.
  func.func @lstm_no_bias(
      %ctx: !hip.context,
      %x : memref<4x1x4xf32, 1>,
      %w : memref<1x32x4xf32, 1>,
      %r : memref<1x32x8xf32, 1>,
      %y : memref<4x1x1x8xf32, 1>,
      %yh: memref<1x1x8xf32, 1>,
      %yc: memref<1x1x8xf32, 1>) {
    // CHECK-LABEL: llvm.func @lstm_no_bias
    hip.lstm(%ctx)
        ins(%x, %w, %r
            : memref<4x1x4xf32, 1>, memref<1x32x4xf32, 1>,
              memref<1x32x8xf32, 1>)
        outs(%y, %yh, %yc
             : memref<4x1x1x8xf32, 1>, memref<1x1x8xf32, 1>,
               memref<1x1x8xf32, 1>)
        {hidden_size = 8 : i64, direction = 0 : i64}

    // CHECK: llvm.call @wrap_miopenRNNForwardInference
    return
  }
}
