// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @gated_rms_norm_sigmoid_f16
// CHECK: llvm.mlir.constant(128 : i64) : i64
// CHECK: %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[ACT:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
// CHECK-NEXT: llvm.call @wrap_gated_rms_norm({{.*}}, %[[ELEM]], %[[ACT]], %[[EPS]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32) -> i32
func.func @gated_rms_norm_sigmoid_f16(%ctx: !hip.context) {
  %input = memref.alloc() : memref<1x2x6144xf16, 1>
  %scale = memref.alloc() : memref<128xf16, 1>
  %gate = memref.alloc() : memref<1x2x6144xf16, 1>
  %output = memref.alloc() : memref<1x2x6144xf16, 1>
  hip.gated_rms_norm(%ctx)
      ins(%input, %scale, %gate : memref<1x2x6144xf16, 1>, memref<128xf16, 1>, memref<1x2x6144xf16, 1>)
      outs(%output : memref<1x2x6144xf16, 1>)
      {activation = "sigmoid", epsilon = 9.99999997E-7 : f32}
  return
}

// CHECK-LABEL: llvm.func @gated_rms_norm_silu_dynamic_f32
// CHECK: llvm.extractvalue %{{.*}}[3, 0]
// CHECK: llvm.mlir.constant(64 : i64) : i64
// CHECK: %[[ELEM2:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: %[[ACT2:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[EPS2:.*]] = llvm.mlir.constant(9.99999974E-6 : f32) : f32
// CHECK-NEXT: llvm.call @wrap_gated_rms_norm({{.*}}, %[[ELEM2]], %[[ACT2]], %[[EPS2]])
func.func @gated_rms_norm_silu_dynamic_f32(%ctx: !hip.context, %n: index) {
  %input = memref.alloc(%n) : memref<?x256xf32, 1>
  %scale = memref.alloc() : memref<64xf32, 1>
  %gate = memref.alloc(%n) : memref<?x256xf32, 1>
  %output = memref.alloc(%n) : memref<?x256xf32, 1>
  hip.gated_rms_norm(%ctx)
      ins(%input, %scale, %gate : memref<?x256xf32, 1>, memref<64xf32, 1>, memref<?x256xf32, 1>)
      outs(%output : memref<?x256xf32, 1>)
      {activation = "silu", epsilon = 1.0e-05 : f32}
  return
}
