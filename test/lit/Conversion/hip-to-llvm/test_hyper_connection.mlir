// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @pre_mix
// CHECK: llvm.call @wrap_hyper_connection_pre_mix({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i32
func.func @pre_mix(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x4x10240xf16, 1>
  %g = memref.alloc() : memref<1x4x10240xf16, 1>
  %y = memref.alloc() : memref<1x4x2560xf16, 1>
  hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : memref<1x4x10240xf16, 1>, memref<1x4x10240xf16, 1>)
      outs(%y : memref<1x4x2560xf16, 1>)
      {branches = 4 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 2560 : i64, reduction_scale = 2.500000e-01 : f32}
  return
}

// --hip-share-matmul-nbits-activation tags reach the runtime as the trailing
// (group, block size) pair.
// CHECK-LABEL: llvm.func @pre_mix_act_quant
// CHECK-DAG: %[[GROUP:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-DAG: %[[BS:.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK: llvm.call @wrap_hyper_connection_pre_mix({{.*}}, %[[GROUP]], %[[BS]])
func.func @pre_mix_act_quant(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x4x10240xf16, 1>
  %g = memref.alloc() : memref<1x4x10240xf16, 1>
  %y = memref.alloc() : memref<1x4x2560xf16, 1>
  hip.hyper_connection_pre_mix(%ctx)
      ins(%x, %g : memref<1x4x10240xf16, 1>, memref<1x4x10240xf16, 1>)
      outs(%y : memref<1x4x2560xf16, 1>)
      {branches = 4 : i64, flattened = 1 : i64, gate_layout = "feature",
       hidden = 2560 : i64, reduction_scale = 2.500000e-01 : f32,
       hip.act_quant_group = 7 : i64, hip.act_quant_block_size = 128 : i64}
  return
}

// CHECK-LABEL: llvm.func @post_mix_norm_act_quant
// CHECK-DAG: %[[GROUP:.*]] = llvm.mlir.constant(9 : i64) : i64
// CHECK-DAG: %[[BS:.*]] = llvm.mlir.constant(256 : i64) : i64
// CHECK: llvm.call @wrap_hyper_connection_post_mix_norm({{.*}}, %[[GROUP]], %[[BS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32, i64, i64) -> i32
func.func @post_mix_norm_act_quant(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x1x10240xf16, 1>
  %b = memref.alloc() : memref<1x1x2560xf16, 1>
  %g = memref.alloc() : memref<1x1x4xf16, 1>
  %w = memref.alloc() : memref<10240xf16, 1>
  %r = memref.alloc() : memref<1x1x10240xf16, 1>
  %n = memref.alloc() : memref<1x1x10240xf16, 1>
  hip.hyper_connection_post_mix_norm(%ctx)
      ins(%x, %b, %g, %w : memref<1x1x10240xf16, 1>, memref<1x1x2560xf16, 1>,
          memref<1x1x4xf16, 1>, memref<10240xf16, 1>)
      outs(%r, %n : memref<1x1x10240xf16, 1>, memref<1x1x10240xf16, 1>)
      {branches = 4 : i64, epsilon = 9.99999997E-7 : f32, flattened = 1 : i64,
       gate_activation = 1 : i64, gate_layout = "branch", hidden = 2560 : i64,
       hip.act_quant_group = 9 : i64, hip.act_quant_block_size = 256 : i64}
  return
}

// The mixer passes its op-state slot, the four inputs and four outputs, the
// element counts and the act projection's M and batch, then the attributes
// and the act-quant tag.
// CHECK-LABEL: llvm.func @mixer
// CHECK-DAG: %[[GROUP:.*]] = llvm.mlir.constant(11 : i64) : i64
// CHECK-DAG: %[[BS:.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK: llvm.call @wrap_hyper_connection_mixer({{.*}}, %[[GROUP]], %[[BS]]) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32, i64, i64) -> i32
func.func @mixer(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x3x10240xf16, 1>
  %p = memref.alloc() : memref<1x3x324xf16, 1>
  %w = memref.alloc() : memref<10240x10x16xui8, 1>
  %s = memref.alloc() : memref<10240x10xf16, 1>
  %y = memref.alloc() : memref<1x3x2560xf16, 1>
  %g = memref.alloc() : memref<1x3x4xf16, 1>
  %a = memref.alloc() : memref<1x3x320xf16, 1>
  %m = memref.alloc() : memref<1x3x10240xf16, 1>
  hip.hyper_connection_mixer(%ctx)
      ins(%x, %p, %w, %s : memref<1x3x10240xf16, 1>, memref<1x3x324xf16, 1>,
          memref<10240x10x16xui8, 1>, memref<10240x10xf16, 1>)
      outs(%y, %g, %a, %m : memref<1x3x2560xf16, 1>, memref<1x3x4xf16, 1>,
           memref<1x3x320xf16, 1>, memref<1x3x10240xf16, 1>)
      {act_scale = 2.500000e-01 : f32, act_width = 320 : i64,
       block_size = 32 : i64, branches = 4 : i64, hidden = 2560 : i64,
       reduction_scale = 2.500000e-01 : f32, swish_alpha = 1.000000e+00 : f32,
       hip.act_quant_group = 11 : i64, hip.act_quant_block_size = 64 : i64}
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
