// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// Without quant_outs the quantized-copy pointers are null. Buffers are
// arguments so the only llvm.mlir.zero ops are those pointers.
// CHECK-LABEL: llvm.func @pre_mix(
// CHECK: %[[N0:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[N1:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: llvm.call @wrap_hyper_connection_pre_mix({{.*}}, %[[N0]], %[[N1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, !llvm.ptr, !llvm.ptr) -> i32
func.func @pre_mix(%ctx: !hip.context, %x: memref<1x4x10240xf16, 1>,
                   %g: memref<1x4x10240xf16, 1>, %y: memref<1x4x2560xf16, 1>) {
  hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : memref<1x4x10240xf16, 1>, memref<1x4x10240xf16, 1>)
      outs(%y : memref<1x4x2560xf16, 1>)
      {branches = 4 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 2560 : i64, reduction_scale = 2.500000e-01 : f32}
  return
}

// CHECK-LABEL: llvm.func @pre_mix_quant(
// CHECK-NOT: llvm.mlir.zero
// CHECK: llvm.call @wrap_hyper_connection_pre_mix({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, !llvm.ptr, !llvm.ptr) -> i32
func.func @pre_mix_quant(%ctx: !hip.context, %x: memref<1x1x10240xf16, 1>,
                         %g: memref<1x1x10240xf16, 1>,
                         %y: memref<1x1x2560xf16, 1>,
                         %q: memref<1x1x2560xi8, 1>,
                         %s: memref<1x1x80xf32, 1>) {
  hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : memref<1x1x10240xf16, 1>, memref<1x1x10240xf16, 1>)
      outs(%y : memref<1x1x2560xf16, 1>)
      quant_outs(%q, %s : memref<1x1x2560xi8, 1>, memref<1x1x80xf32, 1>)
      {branches = 4 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 2560 : i64, reduction_scale = 2.500000e-01 : f32,
       quant_block_size = 32 : i64}
  return
}

// CHECK-LABEL: llvm.func @post_mix_norm_quant
// CHECK: llvm.call @wrap_hyper_connection_post_mix_norm({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, !llvm.ptr, !llvm.ptr) -> i32
func.func @post_mix_norm_quant(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x1x10240xf16, 1>
  %b = memref.alloc() : memref<1x1x2560xf16, 1>
  %g = memref.alloc() : memref<1x1x4xf16, 1>
  %w = memref.alloc() : memref<10240xf16, 1>
  %r = memref.alloc() : memref<1x1x10240xf16, 1>
  %n = memref.alloc() : memref<1x1x10240xf16, 1>
  %q = memref.alloc() : memref<1x1x10240xi8, 1>
  %s = memref.alloc() : memref<1x1x320xf32, 1>
  hip.hyper_connection_post_mix_norm(%ctx)
      ins(%x, %b, %g, %w : memref<1x1x10240xf16, 1>, memref<1x1x2560xf16, 1>,
          memref<1x1x4xf16, 1>, memref<10240xf16, 1>)
      outs(%r, %n : memref<1x1x10240xf16, 1>, memref<1x1x10240xf16, 1>)
      quant_outs(%q, %s : memref<1x1x10240xi8, 1>, memref<1x1x320xf32, 1>)
      {branches = 4 : i64, epsilon = 1.0e-06 : f32, flattened = 1 : i64,
       gate_activation = 1 : i64, gate_layout = "branch", hidden = 2560 : i64,
       quant_block_size = 32 : i64}
  return
}

// CHECK-LABEL: llvm.func @post_mix
// CHECK: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_hyper_connection_post_mix({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
func.func @post_mix(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x4x10240xf16, 1>
  %b = memref.alloc() : memref<1x4x2560xf16, 1>
  %g = memref.alloc() : memref<1x4x4xf16, 1>
  %y = memref.alloc() : memref<1x4x10240xf16, 1>
  hip.hyper_connection_post_mix(%ctx)
      ins(%x, %b, %g : memref<1x4x10240xf16, 1>, memref<1x4x2560xf16, 1>, memref<1x4x4xf16, 1>)
      outs(%y : memref<1x4x10240xf16, 1>)
      {branches = 4 : i64, flattened = 1 : i64, gate_layout = "branch", hidden = 2560 : i64}
  return
}

// CHECK-LABEL: llvm.func @post_mix_stream
// CHECK: llvm.call @wrap_hyper_connection_post_mix
func.func @post_mix_stream(%ctx: !hip.context) {
  %x = memref.alloc() : memref<2x3x4x8xf32, 1>
  %b = memref.alloc() : memref<2x3x8xf32, 1>
  %g = memref.alloc() : memref<2x3x4x8xf32, 1>
  %m = memref.alloc() : memref<2x3x4x4xf32, 1>
  %y = memref.alloc() : memref<2x3x4x8xf32, 1>
  hip.hyper_connection_post_mix(%ctx)
      ins(%x, %b, %g, %m : memref<2x3x4x8xf32, 1>, memref<2x3x8xf32, 1>, memref<2x3x4x8xf32, 1>, memref<2x3x4x4xf32, 1>)
      outs(%y : memref<2x3x4x8xf32, 1>)
      {branches = 4 : i64, flattened = 0 : i64, gate_layout = "feature", hidden = 8 : i64}
  return
}
