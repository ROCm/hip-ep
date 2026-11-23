// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The K-quant projections and token-major GQA call their runtime wrappers with
// bare pointers, the rows (dimension -2) and the folded leading dimensions,
// and formats as the kernels' ids. A module that asks
// for runtime failures to leave the graph gets them marked fallible.
//
// RUN: split-file %s %t
// RUN: hip-mlir-opt --convert-hip-to-llvm %t/calls.mlir | FileCheck %s
// RUN: hip-mlir-opt --convert-hip-to-llvm %t/fallible.mlir | FileCheck %s --check-prefix=FALLIBLE

//--- calls.mlir
// CHECK-LABEL: llvm.func @projections
// CHECK: llvm.call @wrap_quantize_q8_1({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK: llvm.call @wrap_skip_rms_norm_q8_1({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i32
// CHECK: llvm.call @wrap_kquant_matmul_q8_1({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK: llvm.call @wrap_kquant_swiglu_q8_1({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK: llvm.call @wrap_kquant_matmul({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NOT: hip.runtime.fallible
func.func @projections(
    %ctx: !hip.context, %x: memref<1x1x?x512xf32, 1>,
    %skip: memref<1x1x?x512xf32, 1>, %gamma: memref<512xf32, 1>,
    %sum: memref<1x1x?x512xf32, 1>, %q: memref<1x1x?x576xui8, 1>,
    %q4: memref<864xui8, 1>, %q5: memref<1056xui8, 1>,
    %q6: memref<1260xui8, 1>, %y: memref<1x1x?x3xf32, 1>) {
  hip.quantize_q8_1(%ctx) ins(%x : memref<1x1x?x512xf32, 1>)
    outs(%q : memref<1x1x?x576xui8, 1>) {K = 512 : i64}
  hip.skip_rms_norm_q8_1(%ctx)
    ins(%x, %skip, %gamma : memref<1x1x?x512xf32, 1>,
        memref<1x1x?x512xf32, 1>, memref<512xf32, 1>)
    outs(%sum, %q : memref<1x1x?x512xf32, 1>, memref<1x1x?x576xui8, 1>)
    {K = 512 : i64, epsilon = 1.0e-05 : f32}
  hip.kquant_matmul_q8_1(%ctx)
    ins(%q, %q6 : memref<1x1x?x576xui8, 1>, memref<1260xui8, 1>)
    outs(%y : memref<1x1x?x3xf32, 1>)
    {K = 512 : i64, N = 3 : i64, format = "q6_k"}
  hip.kquant_swiglu_q8_1(%ctx)
    ins(%q, %q4, %q6 : memref<1x1x?x576xui8, 1>, memref<864xui8, 1>,
        memref<1260xui8, 1>)
    outs(%y : memref<1x1x?x3xf32, 1>)
    {K = 512 : i64, N = 3 : i64, gate_format = "q4_k", up_format = "q6_k"}
  hip.kquant_matmul(%ctx)
    ins(%x, %q5 : memref<1x1x?x512xf32, 1>, memref<1056xui8, 1>)
    outs(%y : memref<1x1x?x3xf32, 1>)
    {K = 512 : i64, N = 3 : i64, format = "q5_k"}
  return
}

// Without written rows the wrapper gets a null pointer and a zero count.
// CHECK-LABEL: llvm.func @attention
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @wrap_token_major_gqa({{.*}}, %[[NULL]], %[[ZERO]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, !llvm.ptr, i64) -> i32
func.func @attention(
    %ctx: !hip.context, %q: memref<1x?x8x64xf32, 1>,
    %k: memref<1x?x2x64xf16, 1>, %v: memref<1x?x2x64xf16, 1>,
    %mask: memref<1x1x?x?xf16, 1>, %o: memref<1x?x8x64xf32, 1>) {
  hip.token_major_gqa(%ctx)
    ins(%q, %k, %v, %mask : memref<1x?x8x64xf32, 1>, memref<1x?x2x64xf16, 1>,
        memref<1x?x2x64xf16, 1>, memref<1x1x?x?xf16, 1>)
    outs(%o : memref<1x?x8x64xf32, 1>) {scale = 0.125 : f32}
  return
}

// The format's kernel id is the last argument: q5_k is 2.
// CHECK-LABEL: llvm.func @format_id
// CHECK: %[[FORMAT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @wrap_kquant_matmul({{.*}}, %[[FORMAT]])
func.func @format_id(%ctx: !hip.context, %x: memref<4x512xf32, 1>,
                     %w: memref<1056xui8, 1>, %y: memref<4x3xf32, 1>) {
  hip.kquant_matmul(%ctx) ins(%x, %w : memref<4x512xf32, 1>, memref<1056xui8, 1>)
    outs(%y : memref<4x3xf32, 1>) {K = 512 : i64, N = 3 : i64, format = "q5_k"}
  return
}

//--- fallible.mlir
// FALLIBLE-LABEL: llvm.func private @main_graph_internal
// FALLIBLE: %[[STATUS:.*]] = llvm.call @wrap_kquant_matmul({{.*}}) {hip.runtime.fallible}
// FALLIBLE: %[[FAILED:.*]] = llvm.icmp "ne" %[[STATUS]], %{{.*}} : i32
// FALLIBLE: llvm.cond_br %[[FAILED]], ^bb{{[0-9]+}}(%[[STATUS]] : i32), ^bb
module attributes {
  hip.runtime.require_fallible_memref_copies,
  hipdnn.input_count = 3 : i64,
  hipdnn.output_count = 0 : i64,
  hipdnn.input_shapes = [array<i64: 1, -1, 512>, array<i64: 864>,
                         array<i64: 1, -1, 3>],
  hipdnn.output_shapes = []
} {
  func.func @main_graph(%ctx: !hip.context, %x: memref<1x?x512xf32>,
                        %w: memref<864xui8>, %y: memref<1x?x3xf32>) {
    hip.kquant_matmul(%ctx) ins(%x, %w : memref<1x?x512xf32>, memref<864xui8>)
      outs(%y : memref<1x?x3xf32>) {K = 512 : i64, N = 3 : i64, format = "q4_k"}
    return
  }
}
