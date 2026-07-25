// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.global_pool lowers to llvm.call @wrap_global_pool with the
// 8-arg signature
//   (state, input_ptr, output_ptr, outer, reduce_size, data_type, mode, p)
//   -> i32
// (`outer = N*C` and `reduce_size = product(D_i)` are computed off the
// INPUT memref descriptor — so dynamic N / C / D_i all work). The mode and
// p attributes are forwarded as the trailing two i64 args; the call
// signature is identical across modes so the runtime never needs distinct
// LLVM declarations.
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  // Test 1: fully static 4D f32, AVERAGE (mode=0). Both extents are
  // compile-time constants — expect i64 literals, no extractvalue from the
  // descriptor. Also pin the 8-arg call signature.
  func.func @gap_static_nchw_f32(
      %ctx: !hip.context,
      %x: memref<1x3x5x5xf32, 1>,
      %y: memref<1x3x1x1xf32, 1>) {
    // CHECK-LABEL: llvm.func @gap_static_nchw_f32

    hip.global_pool(%ctx) ins(%x : memref<1x3x5x5xf32, 1>)
                          outs(%y : memref<1x3x1x1xf32, 1>)
                          {mode = 0 : i64, p = 2 : i64}

    // CHECK: llvm.mlir.constant(1 : i64)
    // CHECK: llvm.mlir.constant(3 : i64)
    // CHECK: llvm.mlir.constant(5 : i64)
    // CHECK: llvm.mlir.constant(5 : i64)
    // CHECK: llvm.call @wrap_global_pool({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Test 2: dynamic batch + dynamic single spatial dim, rank-3 fp16, MAX
  // (mode=1). Expect `extractvalue %md[3, N]` for each dynamic dim plus
  // `llvm.mul` to fold them into outer / reduce_size i64 args.
  func.func @gmp_dynamic_3d_f16(
      %ctx: !hip.context,
      %x: memref<?x4x?xf16, 1>,
      %y: memref<?x4x1xf16, 1>) {
    // CHECK-LABEL: llvm.func @gmp_dynamic_3d_f16

    hip.global_pool(%ctx) ins(%x : memref<?x4x?xf16, 1>)
                          outs(%y : memref<?x4x1xf16, 1>)
                          {mode = 1 : i64, p = 2 : i64}

    // Dynamic N -> extractvalue at index [3, 0]; static C=4 stays a const.
    // CHECK: llvm.extractvalue %{{.*}}[3, 0]
    // CHECK: llvm.mul
    // Dynamic spatial dim -> extractvalue at index [3, 2].
    // CHECK: llvm.extractvalue %{{.*}}[3, 2]
    // CHECK: llvm.mul
    // CHECK: llvm.call @wrap_global_pool({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }

  // Test 3: GlobalLpPool with p=3 — exercises the `p` attribute through the
  // lowering. mode=2, p=3 must both appear as i64 constants in the call.
  func.func @glp_static_p3_f32(
      %ctx: !hip.context,
      %x: memref<2x4x3x3xf32, 1>,
      %y: memref<2x4x1x1xf32, 1>) {
    // CHECK-LABEL: llvm.func @glp_static_p3_f32

    hip.global_pool(%ctx) ins(%x : memref<2x4x3x3xf32, 1>)
                          outs(%y : memref<2x4x1x1xf32, 1>)
                          {mode = 2 : i64, p = 3 : i64}

    // mode=2 and p=3 are both constants emitted into the call.
    // CHECK: llvm.mlir.constant(2 : i64)
    // CHECK: llvm.mlir.constant(3 : i64)
    // CHECK: llvm.call @wrap_global_pool({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    return
  }
}
