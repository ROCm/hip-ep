// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.nonzero (ONNX NonZero opset 13) lowers to a wrap_nonzero
// runtime call with the expected signature
//   (state, input, output, in_shape, rank, total_elements, k_max, data_type)
// for static-shape inputs.
//
// Covers:
//   - i64 input on a 1-D tensor (Kokoro-style: predicate buffer feeding
//     a transpose+gather chain).
//   - f32 input on a 2-D tensor (general-purpose).
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @nonzero_1d_i64
  // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func @nonzero_1d_i64(%ctx: !hip.context,
                            %input: memref<8xi64, 1>,
                            %output: memref<1x8xi64, 1>) {
    hip.nonzero(%ctx) ins(%input : memref<8xi64, 1>)
                      outs(%output : memref<1x8xi64, 1>)
    return
  }

  // CHECK-LABEL: llvm.func @nonzero_2d_f32
  // CHECK: llvm.call @wrap_nonzero({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func @nonzero_2d_f32(%ctx: !hip.context,
                            %input: memref<3x4xf32, 1>,
                            %output: memref<2x12xi64, 1>) {
    hip.nonzero(%ctx) ins(%input : memref<3x4xf32, 1>)
                      outs(%output : memref<2x12xi64, 1>)
    return
  }
}
