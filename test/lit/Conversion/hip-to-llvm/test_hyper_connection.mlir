// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @pre_mix
// CHECK: llvm.call @wrap_hyper_connection_pre_mix({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, f32) -> i32
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
