// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @engram_gate_two_outputs
// CHECK: llvm.call @wrap_engram_gate({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
func.func @engram_gate_two_outputs(%ctx: !hip.context) {
  %k = memref.alloc() : memref<1x1x4x2560xf16, 1>
  %q = memref.alloc() : memref<1x1x4x2560xf16, 1>
  %v = memref.alloc() : memref<1x1x2560xf16, 1>
  %ks = memref.alloc() : memref<4x2560xf16, 1>
  %qs = memref.alloc() : memref<4x2560xf16, 1>
  %cs = memref.alloc() : memref<4x2560xf16, 1>
  %out = memref.alloc() : memref<1x1x4x2560xf16, 1>
  %normed = memref.alloc() : memref<1x1x4x2560xf16, 1>
  hip.engram_gate(%ctx)
      ins(%k, %q, %v, %ks, %qs, %cs : memref<1x1x4x2560xf16, 1>, memref<1x1x4x2560xf16, 1>, memref<1x1x2560xf16, 1>, memref<4x2560xf16, 1>, memref<4x2560xf16, 1>, memref<4x2560xf16, 1>)
      outs(%out, %normed : memref<1x1x4x2560xf16, 1>, memref<1x1x4x2560xf16, 1>)
      {epsilon = 9.99999997E-7 : f32}
  return
}

// Without the second output, conv_scale and normed are null.
// CHECK-LABEL: llvm.func @engram_gate_one_output
// CHECK-COUNT-2: llvm.mlir.zero : !llvm.ptr
// CHECK: llvm.call @wrap_engram_gate
func.func @engram_gate_one_output(%ctx: !hip.context) {
  %k = memref.alloc() : memref<1x2x2x8xf32, 1>
  %q = memref.alloc() : memref<1x2x2x8xf32, 1>
  %v = memref.alloc() : memref<1x2x8xf32, 1>
  %ks = memref.alloc() : memref<2x8xf32, 1>
  %qs = memref.alloc() : memref<2x8xf32, 1>
  %out = memref.alloc() : memref<1x2x2x8xf32, 1>
  hip.engram_gate(%ctx)
      ins(%k, %q, %v, %ks, %qs : memref<1x2x2x8xf32, 1>, memref<1x2x2x8xf32, 1>, memref<1x2x8xf32, 1>, memref<2x8xf32, 1>, memref<2x8xf32, 1>)
      outs(%out : memref<1x2x2x8xf32, 1>)
      {epsilon = 1.0e-05 : f32}
  return
}
