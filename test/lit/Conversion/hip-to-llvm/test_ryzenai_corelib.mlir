// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --convert-hip-to-llvm="use-ryzenai-corelib=true" routes MatMulNBits,
// RmsNorm and SkipRmsNorm through the RyzenAI CoreLib (NPU) wrapper first and
// falls back to the GPU wrapper only when it returns 1 (not handled).
//
// This test validates:
// - wrap_corelib_* takes the GPU wrapper's arguments plus an i64 data type
// - the GPU wrapper is called in a separate block guarded by rc == 1
// - without the option, only the GPU wrapper is emitted
// ============================================================================

// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm="use-ryzenai-corelib=true" %s | FileCheck %s
// RUN: hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm %s | FileCheck %s --check-prefix=GPU

// CHECK-DAG: llvm.func @wrap_corelib_matmul_nbits(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-DAG: llvm.func @wrap_corelib_rms_norm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, f32, i64, i64) -> i32
// CHECK-DAG: llvm.func @wrap_corelib_skip_rms_norm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
// GPU-NOT: wrap_corelib

module {
  // CHECK-LABEL: llvm.func @corelib_matmul_nbits
  // CHECK: %[[RC:.*]] = llvm.call @wrap_corelib_matmul_nbits(
  // CHECK-NEXT: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: %[[NOT_HANDLED:.*]] = llvm.icmp "eq" %[[RC]], %[[ONE]] : i32
  // CHECK-NEXT: llvm.cond_br %[[NOT_HANDLED]], ^[[FALLBACK:bb[0-9]+]], ^[[CONT:bb[0-9]+]]
  // CHECK-NEXT: ^[[FALLBACK]]:
  // CHECK-NEXT: llvm.call @wrap_matmul_nbits(
  // CHECK-NEXT: llvm.br ^[[CONT]]
  // CHECK-NEXT: ^[[CONT]]:
  // CHECK-NEXT: llvm.return
  // GPU-LABEL: llvm.func @corelib_matmul_nbits
  // GPU: llvm.call @wrap_matmul_nbits(
  func.func @corelib_matmul_nbits(%ctx: !hip.context,
                                  %A: memref<1x128x896xf16, 1>,
                                  %B: memref<896x14x32xui8, 1>,
                                  %scales: memref<896x14xf16, 1>,
                                  %output: memref<1x128x896xf16, 1>) {
    hip.matmul_nbits(%ctx) ins(%A, %B, %scales :
        memref<1x128x896xf16, 1>, memref<896x14x32xui8, 1>,
        memref<896x14xf16, 1>)
        outs(%output : memref<1x128x896xf16, 1>)
        {K = 896 : i64, N = 896 : i64, bits = 4 : i64,
         block_size = 64 : i64, accuracy_level = 4 : i64,
         zp_elem_size = 0 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @corelib_rms_norm
  // CHECK: %[[RC:.*]] = llvm.call @wrap_corelib_rms_norm(
  // CHECK: llvm.icmp "eq" %[[RC]]
  // CHECK: llvm.cond_br
  // CHECK: llvm.call @wrap_rms_norm(
  // GPU-LABEL: llvm.func @corelib_rms_norm
  // GPU: llvm.call @wrap_rms_norm(
  func.func @corelib_rms_norm(%ctx: !hip.context,
                              %input: memref<128x896xf16, 1>,
                              %scale: memref<896xf16, 1>,
                              %output: memref<128x896xf16, 1>) {
    hip.rms_norm(%ctx)
        ins(%input, %scale : memref<128x896xf16, 1>, memref<896xf16, 1>)
        outs(%output : memref<128x896xf16, 1>)
        {axis = -1 : i64, epsilon = 9.99999997e-07 : f32, stash_type = 1 : i64}
    return
  }

  // CHECK-LABEL: llvm.func @corelib_skip_rms_norm
  // CHECK: %[[RC:.*]] = llvm.call @wrap_corelib_skip_rms_norm(
  // CHECK: llvm.icmp "eq" %[[RC]]
  // CHECK: llvm.cond_br
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm(
  // GPU-LABEL: llvm.func @corelib_skip_rms_norm
  // GPU: llvm.call @wrap_skip_simplified_layer_norm(
  func.func @corelib_skip_rms_norm(%ctx: !hip.context,
                                   %input: memref<128x896xf16, 1>,
                                   %skip: memref<128x896xf16, 1>,
                                   %gamma: memref<896xf16, 1>,
                                   %output: memref<128x896xf16, 1>,
                                   %sum: memref<128x896xf16, 1>) {
    hip.skip_rms_norm(%ctx)
        ins(%input, %skip, %gamma : memref<128x896xf16, 1>,
            memref<128x896xf16, 1>, memref<896xf16, 1>)
        outs(%output, %sum : memref<128x896xf16, 1>, memref<128x896xf16, 1>)
        {epsilon = 9.99999997e-07 : f32}
    return
  }
}
