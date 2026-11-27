// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --convert-hip-to-llvm %t/offload.mlir | FileCheck %s
// RUN: not hip-mlir-opt --convert-hip-to-llvm %t/mixed.mlir 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-LABEL: llvm.func @offload
// CHECK:      llvm.mlir.constant(106 : i64) : i64
// CHECK:      %[[W1:.*]] = llvm.mlir.constant(101 : i64) : i64
// CHECK-NEXT: %[[S1:.*]] = llvm.mlir.constant(102 : i64) : i64
// CHECK-NEXT: %[[B1:.*]] = llvm.mlir.constant(103 : i64) : i64
// CHECK-NEXT: %[[W2:.*]] = llvm.mlir.constant(104 : i64) : i64
// CHECK-NEXT: %[[S2:.*]] = llvm.mlir.constant(105 : i64) : i64
// CHECK-NEXT: %[[B2:.*]] = llvm.mlir.constant(106 : i64) : i64
// CHECK-NEXT: %[[NODE:.*]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT: %[[NODES:.*]] = llvm.mlir.constant(13 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_qmoe_offload(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[W1]], %[[S1]], %[[B1]], %[[W2]], %[[S2]], %[[B2]], {{.*}}, %[[NODE]], %[[NODES]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, !llvm.ptr, {{.*}}, f32, f32, i64, i64) -> i32

// ERR: error: 'hip.qmoe' op expert offload needs every fc1/fc2 weight, scale, and bias disk-resident and a hip.expert_offload_node attribute

//--- offload.mlir
func.func @offload(%ctx: !hip.context, %x: memref<1x2x64xf16, 1>,
                   %r: memref<2x4xf16, 1>, %out: memref<1x2x64xf16, 1>) {
  %i1 = arith.constant 101 : i64
  %i2 = arith.constant 102 : i64
  %i3 = arith.constant 103 : i64
  %i4 = arith.constant 104 : i64
  %i5 = arith.constant 105 : i64
  %i6 = arith.constant 106 : i64
  %w1 = hip.get_constant(%ctx, %i1) {hip.disk_resident} : memref<4x64x16xui8, 1>
  %s1 = hip.get_constant(%ctx, %i2) {hip.disk_resident} : memref<4x64x2xf16, 1>
  %b1 = hip.get_constant(%ctx, %i3) {hip.disk_resident} : memref<4x64xf16, 1>
  %w2 = hip.get_constant(%ctx, %i4) {hip.disk_resident} : memref<4x64x8xui8, 1>
  %s2 = hip.get_constant(%ctx, %i5) {hip.disk_resident} : memref<4x64x1xf16, 1>
  %b2 = hip.get_constant(%ctx, %i6) {hip.disk_resident} : memref<4x64xf16, 1>
  hip.qmoe(%ctx) ins(%x, %r, %w1, %s1, %w2, %s2 : memref<1x2x64xf16, 1>, memref<2x4xf16, 1>, memref<4x64x16xui8, 1>, memref<4x64x2xf16, 1>, memref<4x64x8xui8, 1>, memref<4x64x1xf16, 1>)
      fc1_bias(%b1 : memref<4x64xf16, 1>) fc2_bias(%b2 : memref<4x64xf16, 1>)
      outs(%out : memref<1x2x64xf16, 1>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, hip.expert_offload_node = array<i64: 7, 13>, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64}
  return
}

//--- mixed.mlir
func.func @mixed(%ctx: !hip.context, %x: memref<1x2x64xf16, 1>,
                 %r: memref<2x4xf16, 1>, %s1: memref<4x64x2xf16, 1>,
                 %w2: memref<4x64x8xui8, 1>, %s2: memref<4x64x1xf16, 1>,
                 %out: memref<1x2x64xf16, 1>) {
  %i1 = arith.constant 101 : i64
  %w1 = hip.get_constant(%ctx, %i1) {hip.disk_resident} : memref<4x64x16xui8, 1>
  hip.qmoe(%ctx) ins(%x, %r, %w1, %s1, %w2, %s2 : memref<1x2x64xf16, 1>, memref<2x4xf16, 1>, memref<4x64x16xui8, 1>, memref<4x64x2xf16, 1>, memref<4x64x8xui8, 1>, memref<4x64x1xf16, 1>)
      outs(%out : memref<1x2x64xf16, 1>)
      {activation_alpha = 1.0 : f32, activation_beta = 0.0 : f32, activation_type = "swiglu", block_size = 32 : i64, expert_weight_bits = 2 : i64, hip.expert_offload_node = array<i64: 0, 1>, k = 2 : i64, normalize_routing_weights = 1 : i64, swiglu_fusion = 1 : i64, swiglu_limit = 7.0 : f32, use_sparse_mixer = 0 : i64}
  return
}
