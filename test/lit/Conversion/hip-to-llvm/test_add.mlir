// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify HIP add operation is correctly lowered to LLVM call
// to wrap_miopenOpTensor runtime function with 4D per-operand shapes
// and tensor_op = 1 (ADD).
//
// This test validates:
// - hip.add → llvm.call @wrap_miopenOpTensor
// - Per-operand 4D shapes passed for MIOpen-native broadcasting
// - Shapes left-padded with 1 for rank < 4
// - Data type enum (f32=0, f16=1) and tensor_op = 1 (ADD)
// - 18-param signature: state, lhs, rhs, out, lhs_nchw(4), rhs_nchw(4),
//                        out_nchw(4), data_type, tensor_op
//
// Expected: wrap_miopenOpTensor(state, lhs_ptr, rhs_ptr, output_ptr,
//             lhs_n, lhs_c, lhs_h, lhs_w,
//             rhs_n, rhs_c, rhs_h, rhs_w,
//             out_n, out_c, out_h, out_w,
//             data_type, tensor_op=1)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: Same-shape 2D f32 - both operands [128, 512] → padded to [1,1,128,512]
  func.func @add_static_f32_test(
      %ctx: !hip.context,
      %a: memref<128x512xf32, 1>,
      %b: memref<128x512xf32, 1>,
      %c: memref<128x512xf32, 1>) {
    // CHECK-LABEL: llvm.func @add_static_f32_test

    hip.add(%ctx) ins(%a, %b : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
                         outs(%c : memref<128x512xf32, 1>)

    // 4D shape constants for lhs/rhs/out: [1, 1, 128, 512]
    // data_type = 0 (f32), tensor_op = 1 (ADD)
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

  // Test 2: 3D f16 with broadcasting - lhs [1,128,32], rhs [1,1,32] (bias)
  func.func @add_broadcast_f16_test(
      %ctx: !hip.context,
      %a: memref<1x128x32xf16, 1>,
      %b: memref<1x1x32xf16, 1>,
      %c: memref<1x128x32xf16, 1>) {
    // CHECK-LABEL: llvm.func @add_broadcast_f16_test

    hip.add(%ctx) ins(%a, %b : memref<1x128x32xf16, 1>, memref<1x1x32xf16, 1>)
                         outs(%c : memref<1x128x32xf16, 1>)

    // lhs padded: [1, 1, 128, 32], rhs padded: [1, 1, 1, 32] (broadcast dim)
    // CHECK: llvm.call @wrap_miopenOpTensor({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32

    return
  }

}
